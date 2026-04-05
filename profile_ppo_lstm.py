"""
profile_ppo_lstm.py — Microbenchmark the PPO+LSTM training loop.

Instruments each major section with CUDA events (GPU) or perf_counter (CPU),
runs a few iterations, and prints a breakdown table showing where time is spent.

Usage:
    python profile_ppo_lstm.py
"""
import os
import time
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils._pytree import tree_map

from eco_ppo_lstm import (
    Args, EcoAgent, BatchedPlayer, RandomBatchedPlayer, LSTMState,
    make_lstm_state, obs_to_tensor, alloc_obs_buffer,
)
from eco_vec_env import VecSinglePlayerEcoEnv, _stack_obs
from eco_obs_encoder import SinglePlayerEcoEnv
from eco_env import EcoEnv, NUM_ACTIONS


# ── Timer infrastructure ────────────────────────────────────────────────────

class SectionTimer:
    """Accumulates wall-clock time for named sections."""

    def __init__(self):
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._stack: list[tuple[str, float]] = []

    def start(self, name: str):
        torch.cuda.synchronize()
        self._stack.append((name, time.perf_counter()))

    def stop(self, name: str | None = None):
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        pushed_name, t_start = self._stack.pop()
        actual = name or pushed_name
        assert actual == pushed_name, f"Timer mismatch: started {pushed_name}, stopped {actual}"
        self._totals[actual] += (t_end - t_start)
        self._counts[actual] += 1

    def report(self, wall_total: float):
        print()
        print(f"{'Section':<35} {'Total (s)':>10} {'Calls':>7} {'Avg (ms)':>10} {'% of wall':>10}")
        print("─" * 75)
        measured = 0.0
        for name in self._totals:
            total = self._totals[name]
            count = self._counts[name]
            avg_ms = (total / count) * 1000 if count else 0
            pct = total / wall_total * 100 if wall_total > 0 else 0
            measured += total
            print(f"  {name:<33} {total:>10.4f} {count:>7} {avg_ms:>10.3f} {pct:>9.1f}%")
        print("─" * 75)
        print(f"  {'MEASURED':<33} {measured:>10.4f} {'':>7} {'':>10} {measured/wall_total*100:>9.1f}%")
        print(f"  {'WALL CLOCK':<33} {wall_total:>10.4f}")
        unaccounted = wall_total - measured
        print(f"  {'UNACCOUNTED':<33} {unaccounted:>10.4f} {'':>7} {'':>10} {unaccounted/wall_total*100:>9.1f}%")
        print()

        # Verdict
        env_pct = self._totals.get("rollout/env_step", 0) / wall_total * 100
        train_pct = (self._totals.get("train/forward", 0) + self._totals.get("train/backward", 0)) / wall_total * 100
        infer_pct = self._totals.get("rollout/agent_inference", 0) / wall_total * 100
        opp_pct = self._totals.get("rollout/env_step.opponent_nn", 0) / wall_total * 100

        if env_pct > 40:
            print(f"VERDICT: CPU-BOUND — env_step = {env_pct:.1f}% of wall time")
            if opp_pct > 15:
                print(f"  (of which opponent NN = {opp_pct:.1f}% — consider optimizing opponent inference)")
        elif train_pct > 40:
            print(f"VERDICT: GPU-BOUND — train forward+backward = {train_pct:.1f}% of wall time")
        else:
            print(f"VERDICT: BALANCED — env_step={env_pct:.1f}%, inference={infer_pct:.1f}%, train={train_pct:.1f}%")
        print()


def main():
    # Fixed args for profiling
    NUM_ENVS = 128
    NUM_STEPS = 32
    NUM_ITERS = 12      # total iterations
    WARMUP_ITERS = 2    # skip first N for JIT/CUDA warmup
    NUM_PLAYERS = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    agent = EcoAgent(num_players=NUM_PLAYERS).to(device)
    opponent = BatchedPlayer(agent, device, num_envs=NUM_ENVS)
    envs = VecSinglePlayerEcoEnv(
        num_envs=NUM_ENVS, num_players=NUM_PLAYERS,
        opponent=opponent,
        reward_shaping_scale=0, opponent_penalty=0, relative_seat=True,
    )
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    # Allocate buffers
    proto_obs, proto_masks = envs.reset(seed=1)
    obs_buf    = alloc_obs_buffer(proto_obs, NUM_STEPS, NUM_ENVS, device)
    actions    = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.long, device=device)
    logprobs   = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    rewards    = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    dones      = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    values     = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    action_masks = torch.zeros((NUM_STEPS, NUM_ENVS, NUM_ACTIONS), dtype=torch.bool, device=device)

    next_obs   = obs_to_tensor(proto_obs, device)
    next_masks = torch.as_tensor(proto_masks, dtype=torch.bool, device=device)
    next_done  = torch.zeros(NUM_ENVS, device=device)
    next_lstm  = make_lstm_state(agent.lstm_layers, NUM_ENVS, agent.lstm_hidden, device)

    # Wrap opponent.batch_action to measure opponent NN time inside env_step
    timer = SectionTimer()
    _orig_batch_action = opponent.batch_action
    def timed_batch_action(obs_batch, mask_batch, idxs):
        timer.start("rollout/env_step.opponent_nn")
        result = _orig_batch_action(obs_batch, mask_batch, idxs)
        timer.stop()
        return result
    opponent.batch_action = timed_batch_action

    # Wrap env internals to break down env_step cost
    import eco_vec_env
    _orig_stack_obs = eco_vec_env._stack_obs
    def timed_stack_obs(obs_list):
        timer.start("env_step/stack_obs")
        result = _orig_stack_obs(obs_list)
        timer.stop()
        return result
    eco_vec_env._stack_obs = timed_stack_obs

    _orig_encode_for = SinglePlayerEcoEnv._encode_for
    def timed_encode_for(self, player):
        timer.start("env_step/encode_obs")
        result = _orig_encode_for(self, player)
        timer.stop()
        return result
    SinglePlayerEcoEnv._encode_for = timed_encode_for

    _orig_legal_actions = EcoEnv.legal_actions
    def timed_legal_actions(self, state=None):
        timer.start("env_step/legal_actions")
        result = _orig_legal_actions(self, state)
        timer.stop()
        return result
    EcoEnv.legal_actions = timed_legal_actions

    _orig_env_step = EcoEnv.step
    def timed_env_step(self, action):
        timer.start("env_step/game_step")
        result = _orig_env_step(self, action)
        timer.stop()
        return result
    EcoEnv.step = timed_env_step

    _orig_compute_scores = EcoEnv.compute_scores
    def timed_compute_scores(self, state=None):
        timer.start("env_step/compute_scores")
        result = _orig_compute_scores(self, state)
        timer.stop()
        return result
    EcoEnv.compute_scores = timed_compute_scores

    # PPO hyperparams
    gamma = 1.0
    gae_lambda = 0.85
    clip_coef = 0.2
    vf_coef = 1.0
    ent_coef = 0.1
    max_grad_norm = 0.5
    num_minibatches = 4
    update_epochs = 4
    batch_size = NUM_ENVS * NUM_STEPS
    envsperbatch = NUM_ENVS // num_minibatches
    flatinds = np.arange(batch_size).reshape(NUM_STEPS, NUM_ENVS)
    envinds = np.arange(NUM_ENVS)

    print(f"Running {NUM_ITERS} iterations ({WARMUP_ITERS} warmup), "
          f"{NUM_ENVS} envs x {NUM_STEPS} steps = {batch_size} batch")
    print()

    wall_start = None

    for iteration in range(1, NUM_ITERS + 1):
        is_warmup = iteration <= WARMUP_ITERS
        if not is_warmup and wall_start is None:
            # Reset timers after warmup (all method wraps reference `timer` from scope)
            timer = SectionTimer()
            torch.cuda.synchronize()
            wall_start = time.perf_counter()

        initial_lstm = LSTMState(h=next_lstm.h.clone(), c=next_lstm.c.clone())

        # ── ROLLOUT ─────────────────────────────────────────────────────
        for step in range(NUM_STEPS):
            if not is_warmup:
                timer.start("rollout/buffer_store")
            tree_map(lambda buf, val: buf.__setitem__(step, val), obs_buf, next_obs)
            dones[step] = next_done
            action_masks[step] = next_masks
            if not is_warmup:
                timer.stop()

            # Agent inference
            if not is_warmup:
                timer.start("rollout/agent_inference")
            with torch.no_grad():
                action, logprob, _, value, next_lstm = agent.get_action_and_value(
                    next_obs, action_masks[step], next_lstm, next_done
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            if not is_warmup:
                timer.stop()

            # Env step (includes opponent NN via batch_action)
            if not is_warmup:
                timer.start("rollout/env_step")
            next_obs_np, next_masks_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            if not is_warmup:
                timer.stop()

            # Opponent reset
            next_done_np = np.logical_or(terminations, truncations)
            done_indices = list(np.where(next_done_np)[0])
            if not is_warmup:
                timer.start("rollout/opponent_reset")
            if done_indices:
                opponent.reset(done_indices)
            if not is_warmup:
                timer.stop()

            # Data conversion
            if not is_warmup:
                timer.start("rollout/data_conversion")
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            next_obs  = obs_to_tensor(next_obs_np, device)
            next_masks = torch.as_tensor(next_masks_np, dtype=torch.bool, device=device)
            next_done  = torch.Tensor(next_done_np).to(device)
            if not is_warmup:
                timer.stop()

        # ── GAE ─────────────────────────────────────────────────────────
        if not is_warmup:
            timer.start("gae/bootstrap_value")
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm, next_done).reshape(1, -1)
        if not is_warmup:
            timer.stop()

        if not is_warmup:
            timer.start("gae/advantage_calc")
        with torch.no_grad():
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        if not is_warmup:
            timer.stop()

        # ── FLATTEN ─────────────────────────────────────────────────────
        if not is_warmup:
            timer.start("train/flatten")
        b_obs = tree_map(lambda x: x.reshape(-1, *x.shape[2:]), obs_buf)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_action_masks = action_masks.reshape((-1, action_masks.shape[-1]))
        if not is_warmup:
            timer.stop()

        # ── TRAINING ────────────────────────────────────────────────────
        for epoch in range(update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, NUM_ENVS, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                # Forward pass
                if not is_warmup:
                    timer.start("train/forward")
                mb_obs = tree_map(lambda x: x[mb_inds], b_obs)
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    mb_obs, b_action_masks[mb_inds],
                    LSTMState(
                        h=initial_lstm.h[:, mbenvinds],
                        c=initial_lstm.c[:, mbenvinds],
                    ),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                if not is_warmup:
                    timer.stop()

                # Loss computation
                if not is_warmup:
                    timer.start("train/loss")
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                if not is_warmup:
                    timer.stop()

                # Backward + optimize
                if not is_warmup:
                    timer.start("train/backward")
                optimizer.zero_grad()
                loss.backward()
                if not is_warmup:
                    timer.stop()

                if not is_warmup:
                    timer.start("train/grad_clip+step")
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
                if not is_warmup:
                    timer.stop()

        label = "warmup" if is_warmup else f"iter {iteration - WARMUP_ITERS}/{NUM_ITERS - WARMUP_ITERS}"
        print(f"  [{label}] done")

    torch.cuda.synchronize()
    wall_total = time.perf_counter() - wall_start

    measured_iters = NUM_ITERS - WARMUP_ITERS
    sps = (measured_iters * batch_size) / wall_total
    print(f"\n{'='*75}")
    print(f"PPO+LSTM Profiling: {measured_iters} iterations, {NUM_ENVS} envs x {NUM_STEPS} steps")
    print(f"Total wall time: {wall_total:.3f}s | SPS: {sps:.0f}")
    print(f"{'='*75}")

    timer.report(wall_total)
    envs.close()


if __name__ == "__main__":
    main()
