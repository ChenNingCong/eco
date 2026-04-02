"""
eco_gpu_env.py — Single-GPU multi-CPU batched environment for R-öko.

Implements the shared-buffer architecture from hspace.md:
  Layer 0: SinglePlayerEcoEnv.step_gen  (existing, reused as-is)
  Layer 1: EnvWorker — drives N envs, writes obs into shared buffer slots
  Layer 2: GPUBatchedEnv — drives K workers, ONE GPU forward pass per round

Single-node version: workers are driven sequentially by the coordinator
(Python GIL prevents true thread parallelism). The win is batching all
opponent inference into one GPU call across all workers, and zero-copy
writes into a shared pinned buffer.

For true CPU parallelism, swap EnvWorker for a multiprocessing.Process
with shared_memory tensors (see hspace.md for the Ray distributed variant).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Callable

from eco_obs_encoder import (
    SinglePlayerEcoEnv, EcoPyTreeObs, eco_float_dim,
    _make_random_opponent,
    NUM_COLORS, NUM_TYPES, _TOTAL_DECK,
)
from eco_env import NUM_ACTIONS, _STACK


# ── Obs layout: compute field sizes and offsets for flat buffer ──────────────

def _float_field_sizes(num_players: int) -> List[Tuple[str, int]]:
    """Return (field_name, size) for each float field in EcoPyTreeObs order."""
    stack_size = len(_STACK[num_players])
    return [
        ("hands",          num_players * NUM_COLORS * NUM_TYPES),
        ("recycling_side", NUM_COLORS * NUM_TYPES),
        ("waste_side",     NUM_COLORS * NUM_COLORS * NUM_TYPES),
        ("factory_stacks", NUM_COLORS * stack_size),
        ("collected",      num_players * NUM_COLORS * 2),
        ("penalty_pile",   num_players * NUM_COLORS * NUM_TYPES),
        ("scores",         num_players),
        ("draw_pile_size", 1),
        ("draw_pile_comp", NUM_COLORS * NUM_TYPES),
    ]


def _float_field_offsets(num_players: int) -> List[Tuple[str, int, int]]:
    """Return (field_name, start, end) for slicing the flat buffer."""
    offsets = []
    pos = 0
    for name, size in _float_field_sizes(num_players):
        offsets.append((name, pos, pos + size))
        pos += size
    return offsets


# ── Shared Buffer ────────────────────────────────────────────────────────────

class SharedBuffer:
    """Pre-allocated pinned CPU + GPU buffer for zero-copy env→GPU transfer.

    Each env gets one slot. Workers write obs directly into their slot.
    The coordinator DMA's the whole buffer to GPU for one forward pass.
    """

    def __init__(self, max_slots: int, float_dim: int, num_actions: int, device: str):
        self.max_slots = max_slots
        self.float_dim = float_dim
        self.num_actions = num_actions
        self.device = device

        # Pinned CPU memory (writable by workers, DMA-able to GPU)
        self.int_cpu   = torch.zeros(max_slots, 2, dtype=torch.long, pin_memory=True)
        self.float_cpu = torch.zeros(max_slots, float_dim, dtype=torch.float32, pin_memory=True)
        # Default mask: action 0 is legal (prevents NaN for invalid slots in full-buffer forward pass)
        self.masks_cpu = torch.zeros(max_slots, num_actions, dtype=torch.bool, pin_memory=True)
        self.masks_cpu[:, 0] = True
        self.valid_cpu = torch.zeros(max_slots, dtype=torch.bool)

        # GPU mirror
        self.int_gpu   = torch.zeros(max_slots, 2, dtype=torch.long, device=device)
        self.float_gpu = torch.zeros(max_slots, float_dim, dtype=torch.float32, device=device)
        self.masks_gpu = torch.zeros(max_slots, num_actions, dtype=torch.bool, device=device)

        # Action output (GPU → CPU)
        self.actions_gpu = torch.zeros(max_slots, dtype=torch.long, device=device)
        self.actions_cpu = torch.zeros(max_slots, dtype=torch.long, pin_memory=True)

    def upload(self, stream: torch.cuda.Stream):
        """Async DMA: pinned CPU → GPU."""
        with torch.cuda.stream(stream):
            self.int_gpu.copy_(self.int_cpu, non_blocking=True)
            self.float_gpu.copy_(self.float_cpu, non_blocking=True)
            self.masks_gpu.copy_(self.masks_cpu, non_blocking=True)

    def download_actions(self, stream: torch.cuda.Stream):
        """Async DMA: GPU actions → pinned CPU."""
        with torch.cuda.stream(stream):
            self.actions_cpu.copy_(self.actions_gpu, non_blocking=True)

    def clear_valid(self):
        self.valid_cpu.zero_()


def _obs_to_buffer(obs: EcoPyTreeObs, buf: SharedBuffer, slot: int):
    """Write a single EcoPyTreeObs into the shared buffer at the given slot."""
    # Int fields: current_player (1,), phase (1,)
    buf.int_cpu[slot, 0] = int(obs.current_player[0])
    buf.int_cpu[slot, 1] = int(obs.phase[0])
    # Float fields: concatenate all continuous fields
    flat = np.concatenate([
        obs.hands, obs.recycling_side, obs.waste_side,
        obs.factory_stacks, obs.collected,
        obs.penalty_pile, obs.scores, obs.draw_pile_size,
        obs.draw_pile_comp,
    ])
    buf.float_cpu[slot].copy_(torch.from_numpy(flat))
    buf.valid_cpu[slot] = True


def _buffer_to_pytree_obs(buf: SharedBuffer, num_players: int,
                          field_offsets: List[Tuple[str, int, int]]) -> EcoPyTreeObs:
    """Reconstruct a batched EcoPyTreeObs from GPU buffer tensors."""
    fields = {}
    fields["current_player"] = buf.int_gpu[:, 0:1]
    fields["phase"] = buf.int_gpu[:, 1:2]
    for name, start, end in field_offsets:
        fields[name] = buf.float_gpu[:, start:end]
    return EcoPyTreeObs(**fields)


# ── EnvWorker (Layer 1) ─────────────────────────────────────────────────────

class EnvWorker:
    """Drives N SinglePlayerEcoEnvs. Writes opponent obs into shared buffer slots.

    Reuses the existing step_gen() generator from SinglePlayerEcoEnv.
    """

    def __init__(
        self,
        num_envs: int,
        num_players: int,
        slot_offset: int,
        shared_buf: SharedBuffer,
        reward_shaping_scale: float = 0.0,
        opponent_penalty: float = 0.0,
        relative_seat: bool = True,
        seeds: Optional[List[int]] = None,
    ):
        self.num_envs = num_envs
        self.slot_offset = slot_offset
        self.buf = shared_buf

        seeds = seeds or [None] * num_envs
        self.envs: List[SinglePlayerEcoEnv] = [
            SinglePlayerEcoEnv(
                num_players=num_players,
                opponent_fn=None,  # not used — opponents are driven via generators
                seed=s,
                reward_shaping_scale=reward_shaping_scale,
                opponent_penalty=opponent_penalty,
                relative_seat=relative_seat,
            )
            for s in seeds
        ]

        self.gens = [None] * num_envs
        self.pending = set()
        self.results: List[Optional[tuple]] = [None] * num_envs

    def reset(self):
        """Reset all envs. Returns list of (obs, mask) tuples."""
        out = []
        for env in self.envs:
            obs, _ = env.reset()
            mask = env.legal_actions()
            out.append((obs, mask))
        return out

    def prime(self, actions: np.ndarray) -> bool:
        """Start step generators, write first pending obs into buffer.

        Returns True if any env yielded (needs opponent inference).
        """
        self.pending.clear()
        self.results = [None] * self.num_envs

        for i in range(self.num_envs):
            self.gens[i] = self.envs[i].step_gen(int(actions[i]))
            try:
                opp_obs, opp_mask = next(self.gens[i])
                slot = self.slot_offset + i
                _obs_to_buffer(opp_obs, self.buf, slot)
                self.buf.masks_cpu[slot] = torch.from_numpy(opp_mask)
                self.pending.add(i)
            except StopIteration as e:
                self.results[i] = e.value

        return len(self.pending) > 0

    def advance(self) -> bool:
        """Read actions from buffer, send to generators, write next obs.

        Returns True if any env still needs more opponent inference.
        """
        new_pending = set()

        for i in list(self.pending):
            slot = self.slot_offset + i
            action = int(self.buf.actions_cpu[slot].item())
            # Hard enforcement: if GPU returned illegal action, pick first legal
            mask = self.buf.masks_cpu[slot].numpy()
            if not mask[action]:
                legal = np.where(mask)[0]
                action = int(legal[0]) if len(legal) > 0 else 0
            try:
                opp_obs, opp_mask = self.gens[i].send(action)
                _obs_to_buffer(opp_obs, self.buf, slot)
                self.buf.masks_cpu[slot] = torch.from_numpy(opp_mask)
                new_pending.add(i)
            except StopIteration as e:
                self.results[i] = e.value

        self.pending = new_pending
        return len(self.pending) > 0

    def collect_results(self):
        """Return (obs, reward, terminated, info) per env. Auto-resets terminated."""
        out_obs, out_masks = [], []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        infos = []

        for i in range(self.num_envs):
            obs, rew, term, trunc, info = self.results[i]
            if term:
                info["terminal_obs"] = obs
                obs, _ = self.envs[i].reset()
            out_obs.append(obs)
            out_masks.append(self.envs[i].legal_actions())
            rewards[i] = float(rew)
            terminated[i] = term
            infos.append(info)

        return out_obs, out_masks, rewards, terminated, infos


# ── GPUBatchedEnv (Layer 2) ─────────────────────────────────────────────────

class GPUBatchedEnv:
    """Coordinator: drives K EnvWorkers, runs ONE GPU forward pass per round.

    Workers write opponent obs into a shared pinned buffer.
    The coordinator DMA's the buffer to GPU, runs the model, writes actions back.
    Invalid slots (envs that finished early) are wasted FLOPs — cheaper than
    gather/scatter for typical occupancy.
    """

    def __init__(
        self,
        num_workers: int,
        envs_per_worker: int,
        num_players: int,
        model: nn.Module,
        device: str = "cuda",
        reward_shaping_scale: float = 0.0,
        opponent_penalty: float = 0.0,
        relative_seat: bool = True,
        seed: Optional[int] = None,
    ):
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.total_envs = num_workers * envs_per_worker
        self.num_players = num_players
        self.model = model
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

        float_dim = eco_float_dim(num_players)
        self.field_offsets = _float_field_offsets(num_players)
        self.buf = SharedBuffer(self.total_envs, float_dim, NUM_ACTIONS, device)

        # Create workers with contiguous buffer slices
        self.workers: List[EnvWorker] = []
        for k in range(num_workers):
            slot_offset = k * envs_per_worker
            seeds = None
            if seed is not None:
                seeds = [seed + slot_offset + i for i in range(envs_per_worker)]
            worker = EnvWorker(
                num_envs=envs_per_worker,
                num_players=num_players,
                slot_offset=slot_offset,
                shared_buf=self.buf,
                reward_shaping_scale=reward_shaping_scale,
                opponent_penalty=opponent_penalty,
                relative_seat=relative_seat,
                seeds=seeds,
            )
            self.workers.append(worker)

    def reset(self) -> Tuple[EcoPyTreeObs, np.ndarray]:
        """Reset all envs. Returns batched (obs, masks)."""
        all_obs, all_masks = [], []
        for worker in self.workers:
            results = worker.reset()
            for obs, mask in results:
                all_obs.append(obs)
                all_masks.append(mask)
        return _stack_obs(all_obs), np.stack(all_masks)

    @torch.no_grad()
    def _forward_pass(self):
        """DMA buffer to GPU → one forward pass → DMA actions back."""
        self.buf.upload(self.stream)
        self.stream.synchronize()

        # Reconstruct EcoPyTreeObs from flat GPU buffer
        obs_pytree = _buffer_to_pytree_obs(self.buf, self.num_players, self.field_offsets)
        action_mask = self.buf.masks_gpu

        # ONE forward pass on the full buffer
        actions, _, _, _ = self.model.get_action_and_value(obs_pytree, action_mask)
        self.buf.actions_gpu.copy_(actions)

        self.buf.download_actions(self.stream)
        self.stream.synchronize()
        self.buf.clear_valid()

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[EcoPyTreeObs, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """One agent step across all workers.

        Parameters
        ----------
        actions : (total_envs,) int

        Returns
        -------
        obs        : EcoPyTreeObs (batched)
        masks      : (total_envs, NUM_ACTIONS) bool
        rewards    : (total_envs,) float32
        terminated : (total_envs,) bool
        infos      : list of dict
        """
        # Split actions per worker
        chunks = []
        for k, worker in enumerate(self.workers):
            start = k * self.envs_per_worker
            chunks.append(actions[start:start + worker.num_envs])

        # Prime: start generators, write first opponent obs into buffer
        any_pending = False
        for worker, chunk in zip(self.workers, chunks):
            if worker.prime(chunk):
                any_pending = True

        # Drive rounds until all generators are exhausted
        while any_pending:
            self._forward_pass()
            any_pending = False
            for worker in self.workers:
                if worker.advance():
                    any_pending = True

        # Collect results (auto-resets terminated envs)
        all_obs, all_masks = [], []
        all_rewards = []
        all_terminated = []
        all_infos = []
        for worker in self.workers:
            obs_list, mask_list, rewards, terminated, infos = worker.collect_results()
            all_obs.extend(obs_list)
            all_masks.extend(mask_list)
            all_rewards.append(rewards)
            all_terminated.append(terminated)
            all_infos.extend(infos)

        return (
            _stack_obs(all_obs),
            np.stack(all_masks),
            np.concatenate(all_rewards),
            np.concatenate(all_terminated),
            all_infos,
        )

    def close(self):
        pass


def _stack_obs(obs_list: List[EcoPyTreeObs]) -> EcoPyTreeObs:
    """Stack a list of single-env obs into a batched EcoPyTreeObs."""
    return EcoPyTreeObs(**{
        field: np.stack([getattr(o, field) for o in obs_list], axis=0)
        for field in EcoPyTreeObs._fields
    })


# ── Test ─────────────────────────────────────────────────────────────────────

def test_gpu_batched_env():
    """Verify GPUBatchedEnv produces correct shapes and doesn't crash."""
    import sys
    from eco_ppo import EcoAgent

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_players = 3
    num_workers = 4
    envs_per_worker = 8
    total_envs = num_workers * envs_per_worker
    num_steps = 50

    print(f"Testing GPUBatchedEnv: {num_workers} workers × {envs_per_worker} envs "
          f"= {total_envs} total, {num_steps} steps, device={device}")

    # Create model
    model = EcoAgent(num_players=num_players, score_shortcut=False).to(device)
    model.eval()

    # Create GPU batched env
    gpu_env = GPUBatchedEnv(
        num_workers=num_workers,
        envs_per_worker=envs_per_worker,
        num_players=num_players,
        model=model,
        device=device,
        reward_shaping_scale=0.0,
        opponent_penalty=0.0,
        relative_seat=True,
        seed=42,
    )

    # Reset
    obs, masks = gpu_env.reset()
    print(f"  Reset: obs.hands.shape={obs.hands.shape}, masks.shape={masks.shape}")
    assert obs.hands.shape[0] == total_envs, f"Expected batch {total_envs}, got {obs.hands.shape[0]}"
    assert masks.shape == (total_envs, NUM_ACTIONS)

    # Also create a VecSinglePlayerEcoEnv for comparison
    from eco_vec_env import VecSinglePlayerEcoEnv
    vec_env = VecSinglePlayerEcoEnv(
        num_envs=total_envs,
        num_players=num_players,
        seeds=[42 + i for i in range(total_envs)],
        reward_shaping_scale=0.0,
        opponent_penalty=0.0,
        relative_seat=True,
    )
    vec_obs, vec_masks = vec_env.reset()

    # Run steps with random agent actions
    total_rewards_gpu = 0.0
    total_rewards_vec = 0.0
    total_terms_gpu = 0
    total_terms_vec = 0

    def random_actions(masks):
        actions = np.empty(len(masks), dtype=np.int32)
        for i, m in enumerate(masks):
            actions[i] = np.random.choice(np.where(m)[0])
        return actions

    @torch.no_grad()
    def batch_opp_fn(obs_batch, mask_batch):
        """Opponent fn for VecSinglePlayerEcoEnv (for comparison)."""
        from eco_ppo import obs_to_tensor
        obs_t = obs_to_tensor(obs_batch, device)
        mask_t = torch.as_tensor(mask_batch, device=device)
        actions, _, _, _ = model.get_action_and_value(obs_t, mask_t)
        return actions.cpu().numpy()

    for step in range(num_steps):
        # GPU batched env
        agent_actions = random_actions(masks)
        obs, masks, rewards, terminated, infos = gpu_env.step(agent_actions)
        total_rewards_gpu += rewards.sum()
        total_terms_gpu += terminated.sum()

        # Vec env (for comparison)
        vec_actions = random_actions(vec_masks)
        vec_obs, vec_masks, vec_rewards, vec_terminated, _, vec_infos = vec_env.step(
            vec_actions, batch_opp_fn
        )
        total_rewards_vec += vec_rewards.sum()
        total_terms_vec += vec_terminated.sum()

        # Shape checks
        assert obs.hands.shape[0] == total_envs
        assert masks.shape == (total_envs, NUM_ACTIONS)
        assert rewards.shape == (total_envs,)
        assert terminated.shape == (total_envs,)

    print(f"  {num_steps} steps completed successfully")
    print(f"  GPUBatchedEnv: total_reward={total_rewards_gpu:.2f}, "
          f"episodes_done={total_terms_gpu}")
    print(f"  VecEnv:        total_reward={total_rewards_vec:.2f}, "
          f"episodes_done={total_terms_vec}")
    print(f"  (Rewards differ because agent actions and seeds diverge, "
          f"but both should be in similar range)")

    # Sanity: both should have completed some episodes
    assert total_terms_gpu > 0, "No episodes completed in GPUBatchedEnv"
    assert total_terms_vec > 0, "No episodes completed in VecEnv"

    # Reward range check: terminal rewards are +1/-1 + shaping
    # With shaping=0, rewards should be mostly 0 (non-terminal) or ±1 (terminal)
    print(f"  Reward range GPU: [{rewards.min():.2f}, {rewards.max():.2f}]")
    print(f"  Reward range Vec: [{vec_rewards.min():.2f}, {vec_rewards.max():.2f}]")

    print("  All checks passed!")

    # ── Timing comparison ──
    import time

    num_timing_steps = 200
    total_timing_envs = 128

    # GPU Batched
    gpu_env2 = GPUBatchedEnv(
        num_workers=8, envs_per_worker=16,
        num_players=num_players, model=model, device=device,
        seed=123,
    )
    obs2, masks2 = gpu_env2.reset()
    t0 = time.perf_counter()
    for _ in range(num_timing_steps):
        acts = random_actions(masks2)
        obs2, masks2, _, _, _ = gpu_env2.step(acts)
    t_gpu = time.perf_counter() - t0

    # VecEnv
    vec_env2 = VecSinglePlayerEcoEnv(
        num_envs=total_timing_envs, num_players=num_players,
        seeds=[123 + i for i in range(total_timing_envs)],
    )
    vec_obs2, vec_masks2 = vec_env2.reset()
    t0 = time.perf_counter()
    for _ in range(num_timing_steps):
        acts = random_actions(vec_masks2)
        vec_obs2, vec_masks2, _, _, _, _ = vec_env2.step(acts, batch_opp_fn)
    t_vec = time.perf_counter() - t0

    sps_gpu = total_timing_envs * num_timing_steps / t_gpu
    sps_vec = total_timing_envs * num_timing_steps / t_vec
    print(f"\n  Timing ({total_timing_envs} envs, {num_timing_steps} steps):")
    print(f"    GPUBatchedEnv: {t_gpu:.2f}s ({sps_gpu:.0f} steps/s)")
    print(f"    VecEnv:        {t_vec:.2f}s ({sps_vec:.0f} steps/s)")
    print(f"    Ratio: {t_vec/t_gpu:.2f}x")


if __name__ == "__main__":
    test_gpu_batched_env()
