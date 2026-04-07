#!/usr/bin/env python3
"""Benchmark PPO vs DQN head-to-head."""
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark PPO vs DQN")
    parser.add_argument("--ppo-path", required=True)
    parser.add_argument("--dqn-path", required=True)
    parser.add_argument("--ppo-no-lstm", action="store_true")
    parser.add_argument("--ppo-hidden-dim", type=int, default=256)
    parser.add_argument("--dqn-hidden-dim", type=int, default=256)
    parser.add_argument("--num-games", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-discard-draws", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DQN
    q_net = LCQNetwork(hidden_dim=args.dqn_hidden_dim).to(device)
    q_net.load_state_dict(torch.load(args.dqn_path, map_location=device, weights_only=False))
    q_net.eval()
    print(f"Loaded DQN from {args.dqn_path}")

    # Load PPO
    ppo = LCAgent(lstm_hidden=128, no_lstm=args.ppo_no_lstm,
                  hidden_dim=args.ppo_hidden_dim).to(device)
    ppo.load_state_dict(torch.load(args.ppo_path, map_location=device, weights_only=False))
    ppo.eval()
    print(f"Loaded PPO from {args.ppo_path}")

    n_envs = args.num_envs
    factory = LCEnvFactory(new_color_penalty=20, max_lanes=5,
                           max_discard_draws=args.max_discard_draws)

    # ── PPO as agent, DQN as opponent ──
    dqn_opp = DQNBatchedPlayer(q_net, device, n_envs)
    key = key_from_seed(9999)
    envs = VecSinglePlayerEnv(num_envs=n_envs, opponent=dqn_opp,
                              env_factory=factory, key=key)
    obs, masks = envs.reset()
    lstm_state = make_lstm_state(ppo.lstm_layers, n_envs, ppo.lstm_hidden, device)
    done = torch.zeros(n_envs, device=device)

    ppo_wins, dqn_wins, draws = 0, 0, 0
    ppo_scores, dqn_scores = [], []
    total = 0

    while total < args.num_games:
        obs_t = obs_to_tensor(obs, device)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
        with torch.no_grad():
            action, _, _, _, lstm_state = ppo.get_action_and_value(
                obs_t, mask_t, lstm_state, done)
        obs, masks, rewards, terminated, truncated, infos = envs.step(action.cpu().numpy())
        done = torch.as_tensor(terminated, dtype=torch.float32, device=device)
        for i, t in enumerate(terminated):
            if t:
                total += 1
                s = infos[i].get("final_scores")
                seat = infos[i].get("agent_seat", 0)
                if s is not None:
                    ps = float(np.asarray(s)[seat])
                    ds = float(np.asarray(s)[1 - seat])
                    ppo_scores.append(ps)
                    dqn_scores.append(ds)
                    if ps > ds: ppo_wins += 1
                    elif ps < ds: dqn_wins += 1
                    else: draws += 1
                lstm_state.h[:, i] = 0
                lstm_state.c[:, i] = 0
    envs.close()

    print(f"\n{'='*50}")
    print(f"PPO as agent vs DQN as opponent ({total} games):")
    print(f"  PPO wins: {ppo_wins}  DQN wins: {dqn_wins}  Draws: {draws}")
    print(f"  PPO win rate: {ppo_wins/total:.3f}")
    print(f"  PPO score: {np.mean(ppo_scores):.1f} ± {np.std(ppo_scores):.1f}")
    print(f"  DQN score: {np.mean(dqn_scores):.1f} ± {np.std(dqn_scores):.1f}")

    # ── DQN as agent, PPO as opponent ──
    ppo_opp = LSTMBatchedPlayer(ppo, device, n_envs)
    key2 = key_from_seed(8888)
    envs2 = VecSinglePlayerEnv(num_envs=n_envs, opponent=ppo_opp,
                               env_factory=factory, key=key2)
    obs, masks = envs2.reset()
    done2 = torch.zeros(n_envs, device=device)

    dqn_wins2, ppo_wins2, draws2 = 0, 0, 0
    dqn_scores2, ppo_scores2 = [], []
    total2 = 0

    while total2 < args.num_games:
        obs_flat = _flatten_obs_batch(obs)
        obs_t = torch.as_tensor(obs_flat, device=device)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
        with torch.no_grad():
            q_vals = q_net(obs_t)
            q_vals[~mask_t] = -1e8
            actions = q_vals.argmax(dim=1).cpu().numpy()
        obs, masks, rewards, terminated, truncated, infos = envs2.step(actions)
        for i, t in enumerate(terminated):
            if t:
                total2 += 1
                s = infos[i].get("final_scores")
                seat = infos[i].get("agent_seat", 0)
                if s is not None:
                    ds2 = float(np.asarray(s)[seat])
                    ps2 = float(np.asarray(s)[1 - seat])
                    dqn_scores2.append(ds2)
                    ppo_scores2.append(ps2)
                    if ds2 > ps2: dqn_wins2 += 1
                    elif ds2 < ps2: ppo_wins2 += 1
                    else: draws2 += 1
    envs2.close()

    print(f"\nDQN as agent vs PPO as opponent ({total2} games):")
    print(f"  DQN wins: {dqn_wins2}  PPO wins: {ppo_wins2}  Draws: {draws2}")
    print(f"  DQN win rate: {dqn_wins2/total2:.3f}")
    print(f"  DQN score: {np.mean(dqn_scores2):.1f} ± {np.std(dqn_scores2):.1f}")
    print(f"  PPO score: {np.mean(ppo_scores2):.1f} ± {np.std(ppo_scores2):.1f}")

    # Combined
    total_ppo_wins = ppo_wins + ppo_wins2
    total_dqn_wins = dqn_wins + dqn_wins2
    total_draws = draws + draws2
    total_all = total + total2
    print(f"\nCombined ({total_all} games, seat-balanced):")
    print(f"  PPO wins: {total_ppo_wins}  DQN wins: {total_dqn_wins}  Draws: {total_draws}")
    print(f"  PPO win rate: {total_ppo_wins/total_all:.3f}")


if __name__ == "__main__":
    main()
