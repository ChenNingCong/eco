#!/usr/bin/env python3
"""
Train Lost Cities agent using DQN.

Usage:
    python -m game.lc.train_dqn
    python -m game.lc.train_dqn --total-timesteps 50000000 --track
"""
import os
import random
import time

import numpy as np
import torch
import tyro

from abstract import VecSinglePlayerEnv, RandomPlayer, key_from_seed, LSTMBatchedPlayer
from abstract.dqn import DQNTrainer, DQNBatchedPlayer, _flatten_obs_batch, _obs_dim
from game.lc import LCEnvFactory
from game.lc.agent import LCAgent
from game.lc.dqn_agent import LCDQNArgs, LCQNetwork
from game.lc.heuristic_player import HeuristicPlayer


class LCDQNTrainer(DQNTrainer):
    """Lost Cities DQN trainer with benchmark against random."""

    BENCHMARK_ENVS = 32
    BENCHMARK_GAMES = 100

    def __init__(self, config, q_network, target_network, opponent, envs, device):
        super().__init__(config, q_network, target_network, opponent, envs, device)
        self._bench_factory = LCEnvFactory(new_color_penalty=config.new_color_penalty,
                                           max_lanes=config.max_lanes,
                                           max_discard_draws=config.max_discard_draws)
        self._bench_random = RandomPlayer()
        self._bench_heuristic = HeuristicPlayer()
        self._bench_frozen = None  # set externally for best-response/exploitability runs
        self._bench_key = key_from_seed(config.seed + 10000)

    def _run_benchmark(self, opponent, n_games, prefix):
        q_net = self.q_network
        device = self.device
        n_envs = self.BENCHMARK_ENVS

        bench_envs = VecSinglePlayerEnv(
            num_envs=n_envs,
            opponent=opponent,
            env_factory=self._bench_factory,
            key=self._bench_key,
        )

        obs, masks = bench_envs.reset()
        wins, losses, draws = 0, 0, 0
        total = 0
        agent_scores = []
        metrics_accum = {
            "total_points": [], "num_investment": [], "num_played": [],
            "open_lanes": [], "cards_in_hand": [], "discard_draws": [],
        }

        while total < n_games:
            obs_flat = _flatten_obs_batch(obs)
            obs_t = torch.as_tensor(obs_flat, device=device)
            mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                q_vals = q_net(obs_t)
                q_vals[~mask_t] = -1e8
                actions = q_vals.argmax(dim=1).cpu().numpy()

            # Small epsilon to break deterministic cycles in self-play
            for i in range(n_envs):
                if np.random.random() < 0.01:
                    legal = np.where(masks[i])[0]
                    actions[i] = np.random.choice(legal)

            obs, masks, rewards, terminated, truncated, infos = bench_envs.step(actions)
            for i, t in enumerate(terminated):
                if t:
                    total += 1
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
                    gm = infos[i].get("game_metrics")
                    if gm:
                        for k in metrics_accum:
                            if k in gm:
                                metrics_accum[k].append(gm[k])

        bench_envs.close()
        win_rate = wins / total if total else 0
        mean_score = np.mean(agent_scores) if agent_scores else 0.0
        std_score = np.std(agent_scores) if agent_scores else 0.0
        log = {
            f"{prefix}/win_rate": win_rate,
            f"{prefix}/mean_score": mean_score,
            f"{prefix}/std_score": std_score,
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

        # vs random
        log_rand, w_r, l_r, d_r, ms_r, ss_r, ma_r = self._run_benchmark(
            self._bench_random, n_games, "benchmark/vs_random")

        # self-play
        self_opp = DQNBatchedPlayer(self.q_network, self.device, num_envs=self.BENCHMARK_ENVS)
        log_self, w_s, l_s, d_s, ms_s, ss_s, ma_s = self._run_benchmark(
            self_opp, n_games, "benchmark/selfplay")

        # vs heuristic (competent fixed opponent)
        log_heur, w_h, l_h, d_h, ms_h, ss_h, ma_h = self._run_benchmark(
            self._bench_heuristic, n_games, "benchmark/vs_heuristic")

        log = {**log_rand, **log_self, **log_heur, "global_step": global_step}

        # vs frozen PPO (THE exploitability number, for best-response runs)
        if self._bench_frozen is not None:
            log_fz, w_f, l_f, d_f, ms_f, ss_f, ma_f = self._run_benchmark(
                self._bench_frozen, n_games, "benchmark/vs_frozen")
            log.update(log_fz)
            print(f"  vs FROZEN-PPO: {w_f}W/{l_f}L/{d_f}D  win={w_f/max(w_f+l_f+d_f,1)*100:.0f}% "
                  f"| score={ms_f:.1f} inv={np.mean(ma_f['num_investment']):.1f} "
                  f"played={np.mean(ma_f['num_played']):.1f} ddraw={np.mean(ma_f['discard_draws']):.1f}")

        wandb.log(log)

        print(f"  vs Heur:   {w_h}W/{l_h}L/{d_h}D "
              f"| score={ms_h:.1f}±{ss_h:.1f} "
              f"inv={np.mean(ma_h['num_investment']):.1f} "
              f"played={np.mean(ma_h['num_played']):.1f} "
              f"lanes={np.mean(ma_h['open_lanes']):.1f} "
              f"ddraw={np.mean(ma_h['discard_draws']):.1f} "
              f"pts={np.mean(ma_h['total_points']):.0f}")
        print(f"  vs Random: {w_r}W/{l_r}L/{d_r}D "
              f"| score={ms_r:.1f}±{ss_r:.1f} "
              f"inv={np.mean(ma_r['num_investment']):.1f} "
              f"played={np.mean(ma_r['num_played']):.1f} "
              f"lanes={np.mean(ma_r['open_lanes']):.1f} "
              f"ddraw={np.mean(ma_r['discard_draws']):.1f} "
              f"pts={np.mean(ma_r['total_points']):.0f}")
        print(f"  Self-play: {w_s}W/{l_s}L/{d_s}D "
              f"| score={ms_s:.1f}±{ss_s:.1f} "
              f"inv={np.mean(ma_s['num_investment']):.1f} "
              f"played={np.mean(ma_s['num_played']):.1f} "
              f"lanes={np.mean(ma_s['open_lanes']):.1f} "
              f"ddraw={np.mean(ma_s['discard_draws']):.1f} "
              f"pts={np.mean(ma_s['total_points']):.0f}")


def main():
    args = tyro.cli(LCDQNArgs)
    import wandb

    run_name = f"lc__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Q-networks
    q_network = LCQNetwork(hidden_dim=args.hidden_dim).to(device)
    target_network = LCQNetwork(hidden_dim=args.hidden_dim).to(device)
    print(f"Q-network params: {sum(p.numel() for p in q_network.parameters()):,}")

    # Opponent
    frozen_ppo = None
    if args.frozen_ppo_path:
        # Best-response / exploitability: train the DQN against a FIXED frozen PPO agent.
        frozen_ppo = LCAgent(no_lstm=args.frozen_ppo_no_lstm, hidden_dim=args.frozen_ppo_hidden,
                             mlp_depth=args.frozen_ppo_mlp_depth).to(device)
        frozen_ppo.load_state_dict(torch.load(args.frozen_ppo_path, map_location=device, weights_only=False))
        frozen_ppo.eval()
        opponent = LSTMBatchedPlayer(frozen_ppo, device, num_envs=args.num_envs)
        print(f"FROZEN PPO opponent (best-response/exploitability) from {args.frozen_ppo_path}")
    elif args.opponent_mode == "self_play":
        opponent = DQNBatchedPlayer(q_network, device, num_envs=args.num_envs)
    elif args.opponent_mode == "heuristic":
        opponent = HeuristicPlayer()
        print("Heuristic (rule-based) opponent")
    else:
        opponent = RandomPlayer()

    # Env
    factory = LCEnvFactory(new_color_penalty=args.new_color_penalty,
                           score_diff_reward=args.score_diff_reward,
                           zero_one_reward=args.zero_one_reward,
                           raw_score_reward=args.raw_score_reward,
                           dense_reward=args.dense_reward,
                           dense_opponent_delta=args.dense_opponent_delta,
                           dense_opp_penalty=args.dense_opp_penalty,
                           max_lanes=args.max_lanes,
                           max_discard_draws=args.max_discard_draws)
    key = key_from_seed(args.seed)
    envs = VecSinglePlayerEnv(
        num_envs=args.num_envs,
        opponent=opponent,
        env_factory=factory,
        key=key,
    )

    # Train
    trainer = LCDQNTrainer(
        config=args,
        q_network=q_network,
        target_network=target_network,
        opponent=opponent,
        envs=envs,
        device=device,
    )
    if frozen_ppo is not None:
        # benchmark the DQN best-response vs the frozen PPO = the exploitability number
        trainer._bench_frozen = LSTMBatchedPlayer(frozen_ppo, device, num_envs=trainer.BENCHMARK_ENVS)
    trainer.train()


if __name__ == "__main__":
    main()
