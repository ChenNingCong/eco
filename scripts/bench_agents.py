#!/usr/bin/env python3
"""
General head-to-head benchmark for any agent combination.

Supports: dqn (.pt), ppo-lstm (.pkt), ppo-ff (.pkt, --no-lstm)

Usage:
    python -m scripts.bench_agents \
        --p1 dqn:model/lc_dqn_dense_dd5/q_network.pt \
        --p2 ppo-ff:model/lc_50m_ff/latest.pkt \
        --num-games 500

    Agent spec format: TYPE:PATH[:HIDDEN_DIM]
      TYPE: dqn, ppo, ppo-ff
      PATH: checkpoint file
      HIDDEN_DIM: optional, default 256

    Examples:
      dqn:model/lc_dqn_dense_dd5/q_network.pt
      dqn:model/lc_dqn_dense_dd5/q_network.pt:256
      ppo:model/lc_ppo_from_dqn_h512_noent/latest.pkt:512
      ppo-ff:model/lc_50m_ff/latest.pkt
"""
import argparse
import numpy as np
import torch

from abstract import VecSinglePlayerEnv, key_from_seed, LSTMBatchedPlayer
from abstract.dqn import DQNBatchedPlayer, _flatten_obs_batch
from abstract.ppo_lstm import obs_to_tensor, make_lstm_state
from game.lc.engine import LCObs
from game.lc.dqn_agent import LCQNetwork
from game.lc.agent import LCAgent
from game.lc.factory import LCEnvFactory


def parse_agent_spec(spec):
    """Parse 'TYPE:PATH[:HIDDEN_DIM]' into (type, path, hidden_dim)."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Bad agent spec '{spec}', expected TYPE:PATH[:HIDDEN_DIM]")
    atype = parts[0]
    path = parts[1]
    hidden = int(parts[2]) if len(parts) > 2 else 256
    return atype, path, hidden


def load_agent(spec, device):
    """Load agent and return (name, model, type)."""
    atype, path, hidden = parse_agent_spec(spec)
    name = f"{atype}({path.split('/')[-2]}/{path.split('/')[-1]}, h{hidden})"

    if atype == "dqn":
        q_net = LCQNetwork(hidden_dim=hidden).to(device)
        q_net.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        q_net.eval()
        return name, q_net, "dqn", hidden
    elif atype in ("ppo", "ppo-lstm"):
        agent = LCAgent(lstm_hidden=128, no_lstm=False, hidden_dim=hidden).to(device)
        agent.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        agent.eval()
        return name, agent, "ppo", hidden
    elif atype == "ppo-ff":
        agent = LCAgent(lstm_hidden=128, no_lstm=True, hidden_dim=hidden).to(device)
        agent.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        agent.eval()
        return name, agent, "ppo-ff", hidden
    else:
        raise ValueError(f"Unknown agent type '{atype}'. Use: dqn, ppo, ppo-ff")


def make_opponent(model, atype, device, n_envs):
    """Create a batched opponent from a loaded model."""
    if atype == "dqn":
        return DQNBatchedPlayer(model, device, n_envs)
    else:  # ppo or ppo-ff
        return LSTMBatchedPlayer(model, device, n_envs)


def run_matchup(agent_model, agent_type, opp_model, opp_type, device,
                n_envs, n_games, max_discard_draws, seed):
    """Run agent vs opponent, return (wins, losses, draws, agent_scores, opp_scores).

    To avoid short-game bias, each env counts at most games_per_env completions.
    This ensures equal representation regardless of game length."""
    games_per_env = max(1, n_games // n_envs)
    factory = LCEnvFactory(new_color_penalty=20, max_lanes=5,
                           max_discard_draws=max_discard_draws)
    opponent = make_opponent(opp_model, opp_type, device, n_envs)
    key = key_from_seed(seed)
    envs = VecSinglePlayerEnv(num_envs=n_envs, opponent=opponent,
                              env_factory=factory, key=key)
    obs, masks = envs.reset()

    wins, losses, draws, total = 0, 0, 0, 0
    agent_scores, opp_scores = [], []
    env_counts = np.zeros(n_envs, dtype=int)  # completions per env
    envs_done = 0  # number of envs that hit their quota

    # Agent-side state
    if agent_type == "dqn":
        pass  # stateless
    else:
        lstm_state = make_lstm_state(agent_model.lstm_layers, n_envs,
                                     agent_model.lstm_hidden, device)
        done = torch.zeros(n_envs, device=device)

    while envs_done < n_envs:
        if agent_type == "dqn":
            obs_flat = _flatten_obs_batch(obs)
            obs_t = torch.as_tensor(obs_flat, device=device)
            mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                q_vals = agent_model(obs_t)
                q_vals[~mask_t] = -1e8
                actions = q_vals.argmax(dim=1).cpu().numpy()
        else:
            obs_t = obs_to_tensor(obs, device)
            mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                action_t, _, _, _, lstm_state = agent_model.get_action_and_value(
                    obs_t, mask_t, lstm_state, done)
            actions = action_t.cpu().numpy()

        obs, masks, rewards, terminated, truncated, infos = envs.step(actions)

        if agent_type != "dqn":
            done = torch.as_tensor(terminated, dtype=torch.float32, device=device)

        for i, t in enumerate(terminated):
            if t:
                if agent_type != "dqn":
                    lstm_state.h[:, i] = 0
                    lstm_state.c[:, i] = 0
                # Only count if this env hasn't hit its quota
                if env_counts[i] >= games_per_env:
                    continue
                env_counts[i] += 1
                if env_counts[i] == games_per_env:
                    envs_done += 1
                total += 1
                s = infos[i].get("final_scores")
                seat = infos[i].get("agent_seat", 0)
                if s is not None:
                    a_sc = float(np.asarray(s)[seat])
                    o_sc = float(np.asarray(s)[1 - seat])
                    agent_scores.append(a_sc)
                    opp_scores.append(o_sc)
                    if a_sc > o_sc: wins += 1
                    elif a_sc < o_sc: losses += 1
                    else: draws += 1

    envs.close()
    return wins, losses, draws, agent_scores, opp_scores


def main():
    parser = argparse.ArgumentParser(description="Head-to-head agent benchmark")
    parser.add_argument("--p1", required=True, help="Agent 1 spec: TYPE:PATH[:HIDDEN_DIM]")
    parser.add_argument("--p2", required=True, help="Agent 2 spec: TYPE:PATH[:HIDDEN_DIM]")
    parser.add_argument("--num-games", type=int, default=1024,
                        help="Games per seat (total = 2x this)")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-discard-draws", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name1, model1, type1, h1 = load_agent(args.p1, device)
    name2, model2, type2, h2 = load_agent(args.p2, device)
    print(f"P1: {name1}")
    print(f"P2: {name2}")
    print(f"Games per seat: {args.num_games}  max_discard_draws: {args.max_discard_draws}")
    print()

    # P1 as agent, P2 as opponent
    w1, l1, d1, sc1a, sc2a = run_matchup(
        model1, type1, model2, type2, device,
        args.num_envs, args.num_games, args.max_discard_draws, seed=9999)
    t1 = w1 + l1 + d1
    print(f"P1 as agent: {w1}W/{l1}L/{d1}D  "
          f"P1 score={np.mean(sc1a):.1f}±{np.std(sc1a):.1f}  "
          f"P2 score={np.mean(sc2a):.1f}±{np.std(sc2a):.1f}")

    # P2 as agent, P1 as opponent
    w2, l2, d2, sc2b, sc1b = run_matchup(
        model2, type2, model1, type1, device,
        args.num_envs, args.num_games, args.max_discard_draws, seed=8888)
    t2 = w2 + l2 + d2
    print(f"P2 as agent: {w2}W/{l2}L/{d2}D  "
          f"P2 score={np.mean(sc2b):.1f}±{np.std(sc2b):.1f}  "
          f"P1 score={np.mean(sc1b):.1f}±{np.std(sc1b):.1f}")

    # Combined
    total = t1 + t2
    p1_total_wins = w1 + l2  # P1 wins when agent + P1 wins when opponent
    p2_total_wins = l1 + w2
    total_draws = d1 + d2
    all_p1 = sc1a + sc1b
    all_p2 = sc2a + sc2b
    print(f"\nCombined ({total} games, seat-balanced):")
    print(f"  P1 wins: {p1_total_wins}  P2 wins: {p2_total_wins}  Draws: {total_draws}")
    print(f"  P1 win rate: {p1_total_wins/total:.3f}  "
          f"P2 win rate: {p2_total_wins/total:.3f}")
    print(f"  P1 avg score: {np.mean(all_p1):.1f}  P2 avg score: {np.mean(all_p2):.1f}")


if __name__ == "__main__":
    main()
