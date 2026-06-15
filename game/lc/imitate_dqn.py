#!/usr/bin/env python3
"""
Behavioral cloning: train PPO agent to imitate DQN policy.

Phase 1 of DQN→PPO transfer:
1. Collect self-play data from DQN (fresh each epoch to avoid overfitting)
2. Train PPO policy via cross-entropy loss on DQN actions
3. Benchmark the cloned policy with game metrics

Usage:
    python -m game.lc.imitate_dqn \
        --dqn-path model/lc_dqn_dense_dd5/q_network.pt \
        --output model/lc_ppo_from_dqn/pretrained.pkt \
        --num-games 5000 --epochs 50 --batch-size 2048
"""
import argparse
import os
import numpy as np
import torch

from abstract import VecSinglePlayerEnv, RandomPlayer, key_from_seed, LSTMBatchedPlayer
from abstract.dqn import DQNBatchedPlayer, _flatten_obs_batch
from abstract.ppo_lstm import obs_to_tensor, make_lstm_state
from game.lc.engine import LCObs
from game.lc.dqn_agent import LCQNetwork
from game.lc.agent import LCAgent
from game.lc.factory import LCEnvFactory


def collect_data(q_net, device, num_games, num_envs=128, max_discard_draws=5, seed=None):
    """Run DQN self-play, collect (obs_fields, mask, action) tuples.
    Uses a random seed each call when seed=None for fresh data."""
    if seed is None:
        seed = np.random.randint(0, 2**31)
    opponent = DQNBatchedPlayer(q_net, device, num_envs)
    factory = LCEnvFactory(new_color_penalty=20, max_lanes=5, max_discard_draws=max_discard_draws)
    key = key_from_seed(seed)
    envs = VecSinglePlayerEnv(num_envs=num_envs, opponent=opponent, env_factory=factory, key=key)

    obs, masks = envs.reset()
    fields = obs._fields

    obs_bufs = {f: [] for f in fields}
    mask_buf = []
    action_buf = []
    value_buf = []
    games_done = 0
    scores = []

    while games_done < num_games:
        # Greedy DQN action
        obs_flat = _flatten_obs_batch(obs)
        obs_t = torch.as_tensor(obs_flat, device=device)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
        with torch.no_grad():
            q_vals = q_net(obs_t)
            q_vals[~mask_t] = -1e8
            actions = q_vals.argmax(dim=1).cpu().numpy()

        # Store
        for f in fields:
            obs_bufs[f].append(getattr(obs, f).copy())
        mask_buf.append(masks.copy())
        action_buf.append(actions.copy())
        # V(s) = max_a Q(s,a) for greedy policy
        value_buf.append(q_vals.max(dim=1).values.cpu().numpy())

        # Step
        obs, masks, rewards, terminated, truncated, infos = envs.step(actions)
        for i in range(num_envs):
            if terminated[i]:
                games_done += 1
                s = infos[i].get("final_scores")
                seat = infos[i].get("agent_seat", 0)
                if s is not None:
                    scores.append(float(np.asarray(s)[seat]))

    envs.close()

    obs_arrays = {f: np.concatenate(obs_bufs[f]) for f in fields}
    all_masks = np.concatenate(mask_buf)
    all_actions = np.concatenate(action_buf)
    all_values = np.concatenate(value_buf)
    mean_score = np.mean(scores) if scores else 0
    return obs_arrays, all_masks, all_actions, all_values, mean_score, games_done


def benchmark_ppo(agent, device, max_discard_draws=5, num_games=200, num_envs=32):
    """Benchmark PPO agent vs random and self-play, with game metrics."""
    factory = LCEnvFactory(new_color_penalty=20, max_lanes=5, max_discard_draws=max_discard_draws)

    for label, opponent in [("vs_random", RandomPlayer()),
                            ("self_play", LSTMBatchedPlayer(agent, device, num_envs))]:
        key = key_from_seed(12345)
        envs = VecSinglePlayerEnv(num_envs=num_envs, opponent=opponent, env_factory=factory, key=key)
        obs, masks = envs.reset()
        lstm_state = make_lstm_state(agent.lstm_layers, num_envs, agent.lstm_hidden, device)
        done = torch.zeros(num_envs, device=device)

        wins, losses, total = 0, 0, 0
        agent_scores = []
        metrics_accum = {
            "total_points": [], "num_investment": [], "num_played": [],
            "open_lanes": [], "cards_in_hand": [], "discard_draws": [],
        }

        while total < num_games:
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
                    s = infos[i].get("final_scores")
                    seat = infos[i].get("agent_seat", 0)
                    if s is not None:
                        sc = float(np.asarray(s)[seat])
                        agent_scores.append(sc)
                        opp_sc = float(np.asarray(s)[1 - seat])
                        if sc > opp_sc: wins += 1
                        elif sc < opp_sc: losses += 1
                    gm = infos[i].get("game_metrics")
                    if gm:
                        for k in metrics_accum:
                            if k in gm:
                                metrics_accum[k].append(gm[k])
                    lstm_state.h[:, i] = 0
                    lstm_state.c[:, i] = 0

        envs.close()
        mean_sc = np.mean(agent_scores) if agent_scores else 0
        parts = [f"  {label}: {wins}W/{losses}L/{total-wins-losses}D  score={mean_sc:.1f}"]
        for k, vals in metrics_accum.items():
            if vals:
                parts.append(f"{k}={np.mean(vals):.1f}")
        print("  ".join(parts))


def train_bc(agent, q_net, device, epochs=50, batch_size=2048, lr=3e-4,
             max_discard_draws=5, num_games=5000):
    """Behavioral cloning with online data generation (fresh data each epoch).
    Trains both actor (cross-entropy on DQN actions) and critic (MSE on max Q)."""
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    for epoch in range(epochs):
        # Collect fresh data each epoch
        obs_arrays, all_masks, all_actions, all_values, dqn_score, n_games = collect_data(
            q_net, device, num_games, max_discard_draws=max_discard_draws)

        N = len(all_actions)
        fields = list(obs_arrays.keys())

        # Convert to tensors
        obs_tensors = {f: torch.as_tensor(obs_arrays[f], device=device) for f in fields}
        mask_tensor = torch.as_tensor(all_masks, dtype=torch.bool, device=device)
        action_tensor = torch.as_tensor(all_actions, dtype=torch.long, device=device)
        value_tensor = torch.as_tensor(all_values, dtype=torch.float32, device=device)

        perm = np.random.permutation(N)
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_acc = 0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            B = len(idx)

            batch_obs = LCObs(*[obs_tensors[f][idx] for f in fields])
            batch_mask = mask_tensor[idx]
            batch_actions = action_tensor[idx]
            batch_values = value_tensor[idx]

            # Zero LSTM state — no sequential structure during BC
            lstm_state = make_lstm_state(agent.lstm_layers, B, agent.lstm_hidden, device)
            done_t = torch.zeros(B, device=device)

            _, log_prob, entropy, pred_values, _ = agent.get_action_and_value(
                batch_obs, batch_mask, lstm_state, done_t, action=batch_actions)

            actor_loss = -log_prob.mean()  # cross-entropy
            critic_loss = torch.nn.functional.mse_loss(pred_values.squeeze(-1), batch_values)
            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            # Accuracy: greedy match
            with torch.no_grad():
                hidden, _ = agent.get_states(batch_obs, lstm_state, done_t)
                logits = agent._compute_logits(agent.actor_trunk(hidden))
                logits[~batch_mask] = -1e8
                pred = logits.argmax(dim=1)
                acc = (pred == batch_actions).float().mean().item()

            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_acc += acc
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_actor = total_actor_loss / n_batches
        avg_critic = total_critic_loss / n_batches
        avg_acc = total_acc / n_batches
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  "
              f"actor={avg_actor:.4f}  critic={avg_critic:.4f}  acc={avg_acc:.3f}  "
              f"({N} transitions, dqn_score={dqn_score:.1f})")

        # Benchmark every 10 epochs
        if (epoch + 1) % 10 == 0:
            agent.eval()
            benchmark_ppo(agent, device, max_discard_draws=max_discard_draws)
            agent.train()


def main():
    parser = argparse.ArgumentParser(description="Behavioral cloning: DQN → PPO")
    parser.add_argument("--dqn-path", required=True, help="Path to DQN .pt checkpoint")
    parser.add_argument("--dqn-hidden-dim", type=int, default=256)
    parser.add_argument("--output", required=True, help="Output path for PPO .pkt checkpoint")
    parser.add_argument("--num-games", type=int, default=5000,
                        help="Games to collect per epoch from DQN self-play")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=2, help="layers per MLP block (2=original)")
    parser.add_argument("--max-discard-draws", type=int, default=5)
    parser.add_argument("--no-lstm", action="store_true", help="Use feedforward agent (no LSTM)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DQN
    q_net = LCQNetwork(hidden_dim=args.dqn_hidden_dim).to(device)
    q_net.load_state_dict(torch.load(args.dqn_path, map_location=device, weights_only=False))
    q_net.eval()
    print(f"Loaded DQN from {args.dqn_path}")

    # Create PPO agent
    agent = LCAgent(lstm_hidden=args.lstm_hidden, hidden_dim=args.hidden_dim,
                    no_lstm=args.no_lstm, mlp_depth=args.mlp_depth).to(device)
    print(f"PPO agent params: {sum(p.numel() for p in agent.parameters()):,}")

    # Train BC with online data generation
    train_bc(agent, q_net, device,
             epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             max_discard_draws=args.max_discard_draws, num_games=args.num_games)

    # Final benchmark
    agent.eval()
    print("\nFinal benchmark:")
    benchmark_ppo(agent, device, max_discard_draws=args.max_discard_draws)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(agent.state_dict(), args.output)
    print(f"\nSaved pretrained PPO to {args.output}")


if __name__ == "__main__":
    main()
