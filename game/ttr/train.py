#!/usr/bin/env python3
"""
Train Ticket to Ride agent using PPO+LSTM.

Usage:
    python -m game.ttr.train
    python -m game.ttr.train --total-timesteps 50000000 --track
"""
import os
import random
import time

import numpy as np
import torch
import tyro

from abstract import (
    VecSinglePlayerEnv, RandomPlayer, key_from_seed,
    LSTMBatchedPlayer, PPOLSTMTrainer,
)
from abstract.ppo_lstm import obs_to_tensor, make_lstm_state, LSTMState
from game.ttr import TTREnvFactory, TTRAgent, TTRArgs


class TTRTrainer(PPOLSTMTrainer):
    """TTR trainer with benchmark against random opponent + game metrics."""

    BENCHMARK_ENVS = 32
    BENCHMARK_GAMES = 100

    def __init__(self, config: TTRArgs, agent: TTRAgent, opponent, envs, device):
        super().__init__(config, agent, opponent, envs, device)
        self._bench_factory = TTREnvFactory(num_players=config.num_players)
        self._bench_random = RandomPlayer()
        self._bench_key = key_from_seed(config.seed + 10000)

    def _run_benchmark(self, opponent, n_games, prefix, global_step, wandb):
        """Run benchmark games and return logged metrics dict."""
        agent = self.agent
        device = self.device
        n_envs = self.BENCHMARK_ENVS

        bench_envs = VecSinglePlayerEnv(
            num_envs=n_envs,
            opponent=opponent,
            env_factory=self._bench_factory,
            key=self._bench_key,
        )

        obs, masks = bench_envs.reset()
        lstm_state = make_lstm_state(agent.lstm_layers, n_envs, agent.lstm_hidden, device)
        done = torch.zeros(n_envs, device=device)

        wins, losses, draws = 0, 0, 0
        total = 0
        agent_scores = []
        agent_rewards = []
        metrics_accum = {
            "routes_claimed": [], "trains_remaining": [],
            "dest_completed": [], "dest_failed": [], "dest_total": [],
            "route_points": [], "dest_points": [], "dest_penalty": [],
            "avg_route_length": [], "max_route_length": [], "total_points": [],
        }

        while total < n_games:
            obs_t = obs_to_tensor(obs, device)
            mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                action, _, _, _, lstm_state = agent.get_action_and_value(
                    obs_t, mask_t, lstm_state, done)
            obs, masks, rewards, terminated, truncated, infos = bench_envs.step(action.cpu().numpy())
            done = torch.as_tensor(terminated, dtype=torch.float32, device=device)
            for i, t in enumerate(terminated):
                if t:
                    total += 1
                    agent_rewards.append(float(rewards[i]))
                    scores = infos[i].get("final_scores")
                    seat = infos[i].get("agent_seat", 0)
                    if scores is not None:
                        all_scores = np.asarray(scores)
                        agent_scores.append(float(all_scores[seat]))
                        if all_scores[seat] > all_scores.min():
                            wins += 1
                        elif all_scores[seat] < all_scores.max():
                            losses += 1
                        else:
                            draws += 1
                    else:
                        r = float(rewards[i])
                        if r > 0: wins += 1
                        elif r < 0: losses += 1
                        else: draws += 1
                    gm = infos[i].get("game_metrics")
                    if gm:
                        for k in metrics_accum:
                            metrics_accum[k].append(gm[k])
                    lstm_state.h[:, i] = 0
                    lstm_state.c[:, i] = 0

        bench_envs.close()
        win_rate = wins / total if total else 0
        mean_score = np.mean(agent_scores) if agent_scores else 0.0
        std_score = np.std(agent_scores) if agent_scores else 0.0
        mean_reward = np.mean(agent_rewards) if agent_rewards else 0.0
        log = {
            f"{prefix}/win_rate": win_rate,
            f"{prefix}/mean_score": mean_score,
            f"{prefix}/std_score": std_score,
            f"{prefix}/mean_reward": mean_reward,
            f"{prefix}/games": total,
        }
        for k, vals in metrics_accum.items():
            if vals:
                log[f"{prefix}/{k}"] = np.mean(vals)
        return log, wins, losses, draws, mean_score, std_score, metrics_accum

    def benchmark(self, global_step: int):
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        n_games = self.BENCHMARK_GAMES

        # Benchmark vs random
        log_rand, w_r, l_r, d_r, ms_r, ss_r, ma_r = self._run_benchmark(
            self._bench_random, n_games, "benchmark/vs_random", global_step, wandb)

        # Benchmark self-play (agent vs copy of itself)
        self_opponent = LSTMBatchedPlayer(self.agent, self.device, num_envs=self.BENCHMARK_ENVS)
        log_self, w_s, l_s, d_s, ms_s, ss_s, ma_s = self._run_benchmark(
            self_opponent, n_games, "benchmark/selfplay", global_step, wandb)

        log = {**log_rand, **log_self, "global_step": global_step}
        wandb.log(log)

        print(f"  vs Random: {w_r}W/{l_r}L/{d_r}D "
              f"| score={ms_r:.1f}±{ss_r:.1f} "
              f"routes={np.mean(ma_r['routes_claimed']):.1f} "
              f"dest={np.mean(ma_r['dest_completed']):.1f}/{np.mean(ma_r['dest_total']):.1f} "
              f"pts={np.mean(ma_r['total_points']):.0f}")
        print(f"  Self-play: {w_s}W/{l_s}L/{d_s}D "
              f"| score={ms_s:.1f}±{ss_s:.1f} "
              f"routes={np.mean(ma_s['routes_claimed']):.1f} "
              f"dest={np.mean(ma_s['dest_completed']):.1f}/{np.mean(ma_s['dest_total']):.1f} "
              f"pts={np.mean(ma_s['total_points']):.0f}")


def main():
    args = tyro.cli(TTRArgs)
    import wandb

    run_name = f"ttr__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.use_deterministic_algorithms(args.torch_deterministic)
    if args.torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Agent
    agent = TTRAgent(
        num_players=args.num_players,
        lstm_hidden=args.lstm_hidden,
    ).to(device)
    print(f"Agent params: {sum(p.numel() for p in agent.parameters()):,}")

    # Opponent
    if args.opponent_mode == "self_play":
        opponent = LSTMBatchedPlayer(agent, device, num_envs=args.num_envs)
    else:
        opponent = RandomPlayer()

    # Env
    factory = TTREnvFactory(num_players=args.num_players)
    key = key_from_seed(args.seed)
    envs = VecSinglePlayerEnv(
        num_envs=args.num_envs,
        opponent=opponent,
        env_factory=factory,
        key=key,
    )

    # Train
    trainer = TTRTrainer(
        config=args,
        agent=agent,
        opponent=opponent,
        envs=envs,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
