#!/usr/bin/env python3
"""
Train R-Öko agent using PPO+LSTM.

Replicates the best-performing settings (ablation_ff_lstm):
  3 players, self-play, no reward shaping, ent 0.1→0.01, GAE λ=0.85, vf=1.0

Usage:
    python -m game.r_eco.train
    python -m game.r_eco.train --total-timesteps 50000000 --track
"""
import os
import random
import time

import numpy as np
import torch
import tyro

from abstract import (
    VecSinglePlayerEnv, RandomPlayer, key_from_seed,
    PPOConfig, LSTMBatchedPlayer, PPOLSTMTrainer,
)
from abstract import EnvFactory, RewardShaping
from abstract.game import BaseGameEngine
from abstract.ppo_lstm import obs_to_tensor, make_lstm_state, LSTMState
from game.r_eco import RÖkoEnvFactory, RÖkoEngine, EcoAgent, EcoArgs


class RÖkoShapedFactory(EnvFactory):
    """Factory that creates RÖkoEngine wrapped in RewardShaping."""

    def __init__(self, num_players: int = 2,
                 reward_shaping_scale: float = 0.0,
                 opponent_penalty: float = 0.0):
        self.num_players = num_players
        self.reward_shaping_scale = reward_shaping_scale
        self.opponent_penalty = opponent_penalty

    def create(self, rng: np.random.Generator) -> BaseGameEngine:
        engine = RÖkoEngine(rng=rng, num_players=self.num_players)
        if self.reward_shaping_scale > 0:
            engine = RewardShaping(
                engine,
                scale=self.reward_shaping_scale,
                opponent_penalty=self.opponent_penalty,
            )
        return engine


class RÖkoTrainer(PPOLSTMTrainer):
    """R-Öko trainer with benchmark against random opponent."""

    BENCHMARK_ENVS = 32
    BENCHMARK_GAMES = 100

    def __init__(self, config: EcoArgs, agent: EcoAgent, opponent, envs, device):
        super().__init__(config, agent, opponent, envs, device)
        # Create benchmark envs (agent vs random, no reward shaping)
        self._bench_factory = RÖkoEnvFactory(num_players=config.num_players)
        self._bench_random = RandomPlayer()
        self._bench_key = key_from_seed(config.seed + 10000)

    def benchmark(self, global_step: int):
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        agent = self.agent
        device = self.device
        n_envs = self.BENCHMARK_ENVS
        n_games = self.BENCHMARK_GAMES

        bench_envs = VecSinglePlayerEnv(
            num_envs=n_envs,
            opponent=self._bench_random,
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
                    r = float(rewards[i])
                    agent_rewards.append(r)
                    # Extract agent's score from final_scores
                    scores = infos[i].get("final_scores")
                    seat = infos[i].get("agent_seat", 0)
                    if scores is not None:
                        all_scores = np.asarray(scores)
                        agent_scores.append(float(all_scores[seat]))
                        if all_scores[seat] >= all_scores.max():
                            wins += 1
                        else:
                            losses += 1
                    else:
                        if r > 0:
                            wins += 1
                        elif r < 0:
                            losses += 1
                        else:
                            draws += 1
                    # Reset LSTM state for this env
                    lstm_state.h[:, i] = 0
                    lstm_state.c[:, i] = 0

        bench_envs.close()
        win_rate = wins / total
        mean_score = np.mean(agent_scores) if agent_scores else 0.0
        std_score = np.std(agent_scores) if agent_scores else 0.0
        mean_reward = np.mean(agent_rewards)
        std_reward = np.std(agent_rewards)
        print(f"  Benchmark vs random: {wins}W/{losses}L/{draws}D = {win_rate:.1%} "
              f"| score={mean_score:.1f}±{std_score:.1f} reward={mean_reward:+.4f}±{std_reward:.4f}")
        wandb.log({
            "benchmark/win_rate_vs_random": win_rate,
            "benchmark/mean_score": mean_score,
            "benchmark/std_score": std_score,
            "benchmark/mean_reward": mean_reward,
            "benchmark/std_reward": std_reward,
            "benchmark/games": total,
            "global_step": global_step,
        })


def main():
    args = tyro.cli(EcoArgs)
    import wandb

    run_name = f"r_eco__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    agent = EcoAgent(
        num_players=args.num_players,
        lstm_hidden=args.lstm_hidden,
    ).to(device)

    # Opponent
    if args.opponent_mode == "self_play":
        opponent = LSTMBatchedPlayer(agent, device, num_envs=args.num_envs)
    else:
        opponent = RandomPlayer()

    # Env
    factory = RÖkoShapedFactory(
        num_players=args.num_players,
        reward_shaping_scale=args.reward_shaping_scale,
        opponent_penalty=args.opponent_penalty,
    )
    key = key_from_seed(args.seed)
    envs = VecSinglePlayerEnv(
        num_envs=args.num_envs,
        opponent=opponent,
        env_factory=factory,
        key=key,
    )

    # Train
    trainer = RÖkoTrainer(
        config=args,
        agent=agent,
        opponent=opponent,
        envs=envs,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
