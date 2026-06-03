#!/usr/bin/env python3
"""
Benchmark VecSinglePlayerEnv stepping throughput.

Compares:
  1. Generic path (no stepper) — old code path
  2. BatchStepper path (fused JIT) — new optimized path
  3. Full PPO rollout simulation (with NN inference)

Usage:
    python -m scripts.bench_step [--num-envs 2048] [--num-steps 128] [--warmup 5]
"""
import argparse
import time
import numpy as np
import torch

from abstract import VecSinglePlayerEnv, LSTMBatchedPlayer, RandomPlayer, key_from_seed
from abstract.ppo_lstm import obs_to_tensor, make_lstm_state
from game.lc import LCEnvFactory, LCAgent
from game.lc.engine import seed_numba_rng
from game.lc.batch_stepper import LCBatchStepper


def bench_env_step(num_envs, num_steps, warmup, use_stepper):
    """Benchmark pure env stepping (no NN, random actions)."""
    seed_numba_rng(42)
    factory = LCEnvFactory(dense_reward=True, max_discard_draws=50)
    key = key_from_seed(42)
    opponent = RandomPlayer()

    stepper = LCBatchStepper() if use_stepper else None
    envs = VecSinglePlayerEnv(
        num_envs=num_envs, opponent=opponent,
        env_factory=factory, key=key, batch_stepper=stepper,
    )
    obs, masks = envs.reset()

    # Warmup (also JIT-compiles)
    for _ in range(warmup):
        actions = np.random.randint(0, 600, size=num_envs)
        envs.step(actions)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(num_steps):
        actions = np.random.randint(0, 600, size=num_envs)
        envs.step(actions)
    elapsed = time.perf_counter() - t0
    sps = num_envs * num_steps / elapsed
    return sps, elapsed


def bench_ppo_rollout(num_envs, num_steps, warmup, use_stepper, device):
    """Benchmark full PPO rollout (env step + NN inference)."""
    seed_numba_rng(42)
    factory = LCEnvFactory(dense_reward=True, max_discard_draws=50)
    key = key_from_seed(42)

    agent = LCAgent(lstm_hidden=256).to(device)
    agent.eval()
    compiled_agent = torch.compile(agent, dynamic=True)

    opponent = LSTMBatchedPlayer(compiled_agent, device, num_envs=num_envs)

    stepper = LCBatchStepper() if use_stepper else None
    envs = VecSinglePlayerEnv(
        num_envs=num_envs, opponent=opponent,
        env_factory=factory, key=key, batch_stepper=stepper,
    )
    obs_np, masks_np = envs.reset()

    _use_stepper = stepper is not None
    if _use_stepper:
        obs_cls = type(obs_np)
        _obs_cpu = obs_cls(**{
            f: torch.from_numpy(getattr(stepper.obs, f)) for f in obs_cls._fields
        })
        _masks_cpu = torch.from_numpy(stepper.masks)
        _rewards_cpu = torch.from_numpy(stepper._rewards)
        _done_cpu = torch.from_numpy(stepper._done_f32)

        next_obs = obs_cls(**{
            f: getattr(_obs_cpu, f).to(device).clone() for f in obs_cls._fields
        })
        next_masks = _masks_cpu.to(device).clone()
        _obs_pairs = [(getattr(next_obs, f), getattr(_obs_cpu, f)) for f in obs_cls._fields]
    else:
        next_obs = obs_to_tensor(obs_np, device)
        next_masks = torch.as_tensor(masks_np, dtype=torch.bool, device=device)

    next_done = torch.zeros(num_envs, device=device)
    lstm_state = make_lstm_state(agent.lstm_layers, num_envs, agent.lstm_hidden, device)

    def do_rollout():
        nonlocal next_obs, next_masks, next_done, lstm_state
        for step in range(num_steps):
            with torch.no_grad():
                action, _, _, _, lstm_state = compiled_agent.get_action_and_value(
                    next_obs, next_masks, lstm_state, next_done)

            obs_np, masks_np, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            done_np = np.logical_or(term, trunc)
            done_indices = list(np.where(done_np)[0])
            if done_indices:
                opponent.reset(done_indices)

            if _use_stepper:
                for gpu_t, cpu_t in _obs_pairs:
                    gpu_t.copy_(cpu_t)
                next_masks.copy_(_masks_cpu)
                next_done.copy_(_done_cpu)
            else:
                next_obs = obs_to_tensor(obs_np, device)
                next_masks = torch.as_tensor(masks_np, dtype=torch.bool, device=device)
                next_done = torch.Tensor(done_np).to(device)

    # Warmup
    for _ in range(warmup):
        do_rollout()
    torch.cuda.synchronize()

    # Timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    do_rollout()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    sps = num_envs * num_steps / elapsed
    return sps, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=2048)
    parser.add_argument('--num-steps', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Envs: {args.num_envs}, Steps: {args.num_steps}")
    print()

    # 1. Env step only (random actions, no NN)
    print("=== Env Step Only (Random Opponent) ===")
    sps_generic, t_generic = bench_env_step(args.num_envs, args.num_steps, args.warmup, False)
    print(f"  Generic path:      {sps_generic:>10,.0f} SPS  ({t_generic:.3f}s)")
    sps_stepper, t_stepper = bench_env_step(args.num_envs, args.num_steps, args.warmup, True)
    print(f"  BatchStepper path: {sps_stepper:>10,.0f} SPS  ({t_stepper:.3f}s)")
    print(f"  Speedup: {sps_stepper/sps_generic:.2f}x")
    print()

    # 2. Full PPO rollout (with NN)
    print("=== Full PPO Rollout (Self-Play, h256) ===")
    sps_generic, t_generic = bench_ppo_rollout(args.num_envs, args.num_steps, args.warmup, False, device)
    print(f"  Generic path:      {sps_generic:>10,.0f} SPS  ({t_generic:.3f}s)")
    sps_stepper, t_stepper = bench_ppo_rollout(args.num_envs, args.num_steps, args.warmup, True, device)
    print(f"  BatchStepper path: {sps_stepper:>10,.0f} SPS  ({t_stepper:.3f}s)")
    print(f"  Speedup: {sps_stepper/sps_generic:.2f}x")


if __name__ == '__main__':
    main()
