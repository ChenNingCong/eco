#!/usr/bin/env python3
"""Quick benchmark of a TTR checkpoint: agent vs self-play, print detailed scores."""
import sys
import numpy as np
import torch

from abstract import VecSinglePlayerEnv, RandomPlayer, key_from_seed, LSTMBatchedPlayer
from abstract.ppo_lstm import obs_to_tensor, make_lstm_state, LSTMState
from game.ttr import TTREnvFactory, TTRAgent

CKPT = sys.argv[1] if len(sys.argv) > 1 else "model/ttr/ckpt_4997120.pkt"
N_ENVS = 32
N_GAMES = 100

device = "cpu"
agent = TTRAgent(num_players=2).to(device)
agent.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False))
agent.eval()
print(f"Loaded {CKPT}")

factory = TTREnvFactory(num_players=2)  # no shaping — true scores
key = key_from_seed(42)


def run_benchmark(opponent, label):
    envs = VecSinglePlayerEnv(num_envs=N_ENVS, opponent=opponent, env_factory=factory, key=key)
    obs, masks = envs.reset()
    lstm_state = make_lstm_state(agent.lstm_layers, N_ENVS, agent.lstm_hidden, device)
    done = torch.zeros(N_ENVS, device=device)

    wins, losses, draws, total = 0, 0, 0, 0
    agent_scores, opp_scores = [], []
    metrics = {
        "routes_claimed": [], "trains_remaining": [],
        "dest_completed": [], "dest_failed": [], "dest_total": [],
        "route_points": [], "dest_points": [], "dest_penalty": [],
        "total_points": [],
    }

    while total < N_GAMES:
        obs_t = obs_to_tensor(obs, device)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
        with torch.no_grad():
            action, _, _, _, lstm_state = agent.get_action_and_value(
                obs_t, mask_t, lstm_state, done)
        obs, masks, rewards, terminated, truncated, infos = envs.step(action.cpu().numpy())
        done = torch.as_tensor(terminated, dtype=torch.float32, device=device)
        for i, t in enumerate(terminated):
            if t:
                total += 1
                scores = infos[i].get("final_scores")
                seat = infos[i].get("agent_seat", 0)
                if scores is not None:
                    all_s = np.asarray(scores)
                    agent_scores.append(float(all_s[seat]))
                    opp_seat = 1 - seat
                    opp_scores.append(float(all_s[opp_seat]))
                    if all_s[seat] > all_s.min():
                        wins += 1
                    elif all_s[seat] < all_s.max():
                        losses += 1
                    else:
                        draws += 1
                gm = infos[i].get("game_metrics")
                if gm:
                    for k in metrics:
                        if k in gm:
                            metrics[k].append(gm[k])
                lstm_state.h[:, i] = 0
                lstm_state.c[:, i] = 0

    envs.close()

    print(f"\n=== {label} ({total} games) ===")
    print(f"  W/L/D: {wins}/{losses}/{draws}  winrate={wins/total:.2%}")
    print(f"  Agent score:  mean={np.mean(agent_scores):.1f}  std={np.std(agent_scores):.1f}  min={np.min(agent_scores):.0f}  max={np.max(agent_scores):.0f}")
    print(f"  Opponent score: mean={np.mean(opp_scores):.1f}  std={np.std(opp_scores):.1f}")
    for k, vals in metrics.items():
        if vals:
            print(f"  {k}: {np.mean(vals):.2f}")

    # Print a few individual game details
    print(f"\n  Sample scores (agent, opp): ", end="")
    for j in range(min(10, len(agent_scores))):
        print(f"({agent_scores[j]:.0f},{opp_scores[j]:.0f}) ", end="")
    print()


# Self-play benchmark
self_opp = LSTMBatchedPlayer(agent, device, num_envs=N_ENVS)
run_benchmark(self_opp, "Self-play")

# vs Random
run_benchmark(RandomPlayer(), "vs Random")
