#!/usr/bin/env python3
"""Head-to-head win-rate between two frozen LCAgent (PPO) checkpoints.

Runs BOTH directions (A-as-actor vs B-opponent, then B-as-actor vs A-opponent)
to cancel any asymmetry between the get_action_and_value path and the
LSTMBatchedPlayer path, and reports A's overall win-rate. Seats are already
randomized inside VecSinglePlayerEnv (we read agent_seat from info).

Usage:
  python -m game.lc.eval_h2h --ckpt-a model/_h2h_1b_snapshot.pkt \
      --ckpt-b model/lc_ppo_dqninit_final50m/latest.pkt --games 3000
"""
import argparse
import numpy as np
import torch

from abstract import VecSinglePlayerEnv, key_from_seed, LSTMBatchedPlayer
from abstract.ppo_lstm import obs_to_tensor, make_lstm_state
from game.lc import LCEnvFactory, LCAgent


def load_agent(path, device, hidden_dim, no_lstm):
    a = LCAgent(no_lstm=no_lstm, hidden_dim=hidden_dim).to(device)
    a.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    a.eval()
    return a


def run_dir(actor, opp_model, device, n_games, n_envs, factory, key):
    """Play n_games with `actor` in the learner seat vs `opp_model` opponent.
    Returns (wins, losses, draws, actor_scores, opp_scores)."""
    opponent = LSTMBatchedPlayer(opp_model, device, num_envs=n_envs)
    envs = VecSinglePlayerEnv(num_envs=n_envs, opponent=opponent,
                             env_factory=factory, key=key)
    obs, masks = envs.reset()
    lstm_state = make_lstm_state(actor.lstm_layers, n_envs, actor.lstm_hidden, device)
    done = torch.zeros(n_envs, device=device)
    wins = losses = draws = total = 0
    a_scores, o_scores = [], []
    while total < n_games:
        obs_t = obs_to_tensor(obs, device)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
        with torch.no_grad():
            action, _, _, _, lstm_state = actor.get_action_and_value(
                obs_t, mask_t, lstm_state, done)
        obs, masks, rewards, terminated, truncated, infos = envs.step(action.cpu().numpy())
        done = torch.as_tensor(terminated, dtype=torch.float32, device=device)
        for i, t in enumerate(terminated):
            if t:
                total += 1
                sc = infos[i].get("final_scores")
                seat = infos[i].get("agent_seat", 0)
                if sc is not None:
                    sc = np.asarray(sc)
                    a_scores.append(float(sc[seat]))
                    o_scores.append(float(sc[1 - seat]))
                    if sc[seat] > sc.min():
                        wins += 1
                    elif sc[seat] < sc.max():
                        losses += 1
                    else:
                        draws += 1
                lstm_state.h[:, i] = 0
                lstm_state.c[:, i] = 0
    envs.close()
    return wins, losses, draws, a_scores, o_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a", required=True)
    p.add_argument("--ckpt-b", required=True)
    p.add_argument("--games", type=int, default=3000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--no-lstm", action="store_true", default=True)
    p.add_argument("--max-discard-draws", type=int, default=5)
    p.add_argument("--seed", type=int, default=20000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = load_agent(args.ckpt_a, device, args.hidden_dim, args.no_lstm)
    B = load_agent(args.ckpt_b, device, args.hidden_dim, args.no_lstm)
    factory = LCEnvFactory(new_color_penalty=20, max_discard_draws=args.max_discard_draws)
    print(f"A = {args.ckpt_a}")
    print(f"B = {args.ckpt_b}")
    print(f"games per direction = {args.games}, num_envs = {args.num_envs}\n")

    half = args.games
    # Direction 1: A is the actor
    w1, l1, d1, a1, o1 = run_dir(A, B, device, half, args.num_envs, factory,
                                 key_from_seed(args.seed))
    # Direction 2: B is the actor (so A is the opponent); A wins = B's losses
    w2, l2, d2, a2, o2 = run_dir(B, A, device, half, args.num_envs, factory,
                                 key_from_seed(args.seed + 777))

    print(f"Dir1 (A actor): A {w1}W/{l1}L/{d1}D  win={w1/(w1+l1+d1)*100:.1f}%  "
          f"A_score={np.mean(a1):.1f}  B_score={np.mean(o1):.1f}")
    print(f"Dir2 (B actor): B {w2}W/{l2}L/{d2}D  B_win={w2/(w2+l2+d2)*100:.1f}%  "
          f"B_score={np.mean(a2):.1f}  A_score={np.mean(o2):.1f}")

    # Combined A win-rate over all 2*half games (A wins in dir2 = l2)
    A_wins = w1 + l2
    A_loss = l1 + w2
    A_draw = d1 + d2
    tot = A_wins + A_loss + A_draw
    A_scores = a1 + o2
    B_scores = o1 + a2
    print("\n==== COMBINED (both directions, seat- & path-balanced) ====")
    print(f"A wins {A_wins}/{tot} = {A_wins/tot*100:.1f}%  | "
          f"A losses {A_loss/tot*100:.1f}%  draws {A_draw/tot*100:.1f}%")
    # decisive vs draws: win-rate among decided games
    dec = A_wins + A_loss
    print(f"A win-rate among DECIDED games = {A_wins/dec*100:.1f}%")
    print(f"A mean score = {np.mean(A_scores):.2f} ± {np.std(A_scores):.1f}  | "
          f"B mean score = {np.mean(B_scores):.2f} ± {np.std(B_scores):.1f}")


if __name__ == "__main__":
    main()
