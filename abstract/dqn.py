# abstract/dqn.py — Game-agnostic DQN training for turn-based games.
#   - ReplayBuffer: flat numpy buffer for NamedTuple observations
#   - DQNTrainer: multi-env DQN with masked ε-greedy and target network
#
# Based on CleanRL's dqn_atari.py, adapted for:
#   - NamedTuple observations (pytree flatten/unflatten)
#   - Action masks (illegal action masking in ε-greedy and Q-target)
#   - VecSinglePlayerEnv interface
#   - Self-play opponent support

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, NamedTuple
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils._pytree import tree_map

from .ppo_lstm import layer_init, obs_to_tensor
from .player import BasePlayer, SlicedPlayer


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class DQNConfig:
    """Game-agnostic DQN hyperparameters."""
    exp_name: str = "dqn"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ppo-eco"
    wandb_entity: Optional[str] = None
    total_timesteps: int = 50_000_000
    learning_rate: float = 1e-4
    num_envs: int = 128
    buffer_size: int = 1_000_000
    """replay buffer size (total transitions, not per-env)"""
    gamma: float = 1.0
    tau: float = 1.0
    """target network update rate (1.0 = hard copy)"""
    double_dqn: bool = False
    """Double DQN: pick next action with online net, evaluate with target net (anti-overestimation)"""
    huber_loss: bool = False
    """use Huber/smooth-L1 loss instead of MSE (robust to TD-error spikes)"""
    # ── Rainbow component toggles (default off → identical to vanilla DQN) ──
    n_step: int = 1
    """multi-step returns: accumulate n rewards before bootstrapping (1 = vanilla 1-step)"""
    tree_backup: bool = False
    """Retrace-family off-policy correction for n-step. With a greedy deterministic
    target, Retrace(λ=1) reduces to tree-backup: cut the n-step trace at the first
    exploratory (non-greedy) action and bootstrap there. Removes uncorrected-n-step
    bias from exploration; EXACT for gamma=1 (bootstrap discount = 1 at any cut
    length). No effect when n_step == 1."""
    distributional: bool = False
    """C51 distributional RL: predict a return distribution + cross-entropy loss"""
    n_atoms: int = 51
    """number of atoms in the C51 return distribution"""
    v_min: float = -10.0
    """lower bound of the C51 support"""
    v_max: float = 10.0
    """upper bound of the C51 support"""
    prioritized: bool = False
    """prioritized experience replay (simplest vectorized-proportional impl)"""
    per_alpha: float = 0.5
    """PER priority exponent"""
    per_beta: float = 0.4
    """PER importance-sampling exponent (annealed to 1 over training)"""
    target_network_frequency: int = 1000
    """update target network every N steps (in env steps, not gradient steps)"""
    batch_size: int = 256
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.10
    """fraction of total_timesteps for ε annealing"""
    learning_starts: int = 10000
    """start training after this many env steps"""
    train_frequency: int = 4
    """train every N env steps"""
    grad_steps_per_train: int = 1
    """gradient steps per training call"""
    max_grad_norm: float = 10.0
    model_dir: str = "model"
    log_interval: int = 81_920
    save_interval: int = 81_920
    hidden_dim: int = 256


# ── Replay Buffer ───────────────────────────────────────────────────────────

def _flatten_obs(obs) -> np.ndarray:
    """Flatten a NamedTuple obs into a single float32 vector.
    Integer fields (like phase) are cast to float32."""
    parts = []
    for field in obs._fields:
        v = getattr(obs, field)
        parts.append(v.astype(np.float32).ravel())
    return np.concatenate(parts)


def _flatten_obs_batch(obs) -> np.ndarray:
    """Flatten batched NamedTuple obs (B, ...) into (B, D) float32."""
    parts = []
    for field in obs._fields:
        v = getattr(obs, field)
        parts.append(v.astype(np.float32).reshape(v.shape[0], -1))
    return np.concatenate(parts, axis=1)


def _obs_dim(obs) -> int:
    """Get total flat dimension from a batched obs."""
    total = 0
    for field in obs._fields:
        v = getattr(obs, field)
        total += int(np.prod(v.shape[1:]))
    return total


class ReplayBuffer:
    """Simple replay buffer storing flattened obs + masks."""

    def __init__(self, buffer_size: int, obs_dim: int, num_actions: int, device,
                 n_step: int = 1, gamma: float = 1.0, num_envs: int = 1,
                 prioritized: bool = False, alpha: float = 0.5, tree_backup: bool = False):
        self.buffer_size = buffer_size
        self.device = device
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.masks = np.zeros((buffer_size, num_actions), dtype=np.bool_)
        self.next_masks = np.zeros((buffer_size, num_actions), dtype=np.bool_)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.pos = 0
        self.full = False
        # ── Rainbow: multi-step returns (one n-step deque per env) ──
        self.n_step = n_step
        self.gamma = gamma
        self.tree_backup = tree_backup
        self.num_envs = num_envs
        # tree_backup stores RAW 1-step transitions (no eager collapse); its n-step
        # target is recomputed at train time vs the current net (sample-time cut).
        self._deques = [deque(maxlen=n_step) for _ in range(num_envs)] \
            if (n_step > 1 and not tree_backup) else None
        # ── Rainbow: prioritized replay (simplest method = vectorized proportional,
        #    no segment tree; new transitions inserted at max priority) ──
        self.prioritized = prioritized
        self.alpha = alpha
        self.priorities = np.zeros(buffer_size, dtype=np.float32) if prioritized else None
        self.max_priority = 1.0

    def add(self, obs, next_obs, mask, next_mask, action, reward, done):
        """Add a single transition."""
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.masks[self.pos] = mask
        self.next_masks[self.pos] = next_mask
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        if self.prioritized:
            self.priorities[self.pos] = self.max_priority   # new sample → max priority
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def _n_step_info(self, dq):
        """Collapse an n-step deque into (reward_n, next_obs, next_mask, done)."""
        reward, next_obs, next_mask, done = 0.0, dq[-1][1], dq[-1][3], dq[-1][6]
        for i in range(len(dq)):
            if self.tree_backup and i >= 1 and dq[i][7]:
                # behavior went off-greedy at step i → cut the trace, bootstrap at
                # s_i (= obs/mask of transition i), accumulating only r_0..r_{i-1}.
                next_obs, next_mask, done = dq[i][0], dq[i][2], False
                break
            reward += self.gamma**i * dq[i][5]
            if dq[i][6]:   # episode ended within the window → stop here
                next_obs, next_mask, done = dq[i][1], dq[i][3], True
                break
        return reward, next_obs, next_mask, done

    def add_batch(self, obs, next_obs, masks, next_masks, actions, rewards, dones, explore=None):
        """Add a batch of transitions. obs/next_obs are (N, D) flat arrays.
        `explore` (N,) bool flags exploratory actions, used by tree-backup."""
        N = len(actions)
        # ── Rainbow: multi-step path — accumulate per-env, store collapsed n-step ──
        if self._deques is not None:
            for e in range(N):
                dq = self._deques[e]
                expl = bool(explore[e]) if explore is not None else False
                dq.append((obs[e], next_obs[e], masks[e], next_masks[e],
                           int(actions[e]), float(rewards[e]), bool(dones[e]), expl))
                if len(dq) < self.n_step:
                    continue
                r_n, no, nm, dn = self._n_step_info(dq)
                self.add(dq[0][0], no, dq[0][2], nm, dq[0][4], r_n, float(dn))
                if dones[e]:
                    # HANDOFF §4 fix: drain (don't clear) so the last n-1
                    # end-of-episode windows — incl. the terminal transition —
                    # are not dropped. Deliberate deviation from cleanrl
                    # rainbow_atari (harmless in long Atari eps, harmful in
                    # LC's ~25-step eps).
                    if len(dq) == self.n_step:
                        dq.popleft()            # this window already emitted above
                    while dq:                   # flush the shorter end-of-episode windows
                        r_n, no, nm, dn = self._n_step_info(dq)
                        self.add(dq[0][0], no, dq[0][2], nm, dq[0][4], r_n, float(dn))
                        dq.popleft()
            return
        # ── 1-step fast path (unchanged) ──
        if self.pos + N <= self.buffer_size:
            sl = slice(self.pos, self.pos + N)
            self.obs[sl] = obs
            self.next_obs[sl] = next_obs
            self.masks[sl] = masks
            self.next_masks[sl] = next_masks
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.dones[sl] = dones
            if self.prioritized:
                self.priorities[sl] = self.max_priority
            self.pos += N
            if self.pos >= self.buffer_size:
                self.full = True
                self.pos = self.pos % self.buffer_size
        else:
            # Wrap around
            for i in range(N):
                self.add(obs[i], next_obs[i], masks[i], next_masks[i],
                         actions[i], rewards[i], dones[i])

    def sample(self, batch_size: int, beta: float = 0.4):
        upper = self.buffer_size if self.full else self.pos
        if self.prioritized:
            # vectorized proportional sampling (no segment tree); IS weights for the loss
            probs = self.priorities[:upper] ** self.alpha
            probs = probs / probs.sum()
            idx = np.random.choice(upper, size=batch_size, p=probs)
            weights = (upper * probs[idx]) ** (-beta)
            weights = (weights / weights.max()).astype(np.float32)
        else:
            idx = np.random.randint(0, upper, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.masks[idx], dtype=torch.bool, device=self.device),
            torch.as_tensor(self.next_masks[idx], dtype=torch.bool, device=self.device),
            torch.as_tensor(self.actions[idx], dtype=torch.long, device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
            torch.as_tensor(weights, device=self.device),
            idx,
        )

    def update_priorities(self, idx, td_errors):
        """Set sampled transitions' priorities to |TD error| (+eps). PER only."""
        if not self.prioritized:
            return
        p = np.abs(td_errors) + 1e-6
        self.priorities[idx] = p
        self.max_priority = max(self.max_priority, float(p.max()))

    def size(self):
        return self.buffer_size if self.full else self.pos


# ── Abstract Q-Network ──────────────────────────────────────────────────────

class BaseQNetwork(nn.Module, ABC):
    """Abstract Q-network. Takes flat obs tensor, returns Q-values for all actions."""

    @abstractmethod
    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """(B, obs_dim) -> (B, num_actions)"""
        ...

    @property
    @abstractmethod
    def num_actions(self) -> int:
        ...


# ── DQN Player (for self-play opponent) ─────────────────────────────────────

class DQNBatchedPlayer(BasePlayer):
    """Greedy Q-network opponent for self-play. Stateless (no LSTM)."""

    def __init__(self, q_network: BaseQNetwork, device, num_envs: int):
        self.q_network = q_network
        self.device = device
        self.num_envs = num_envs

    def reset(self, env_indices=None):
        pass  # stateless

    def batch_action(self, obs_batch, mask_batch: np.ndarray, idxs: list) -> np.ndarray:
        obs_flat = _flatten_obs_batch(obs_batch)
        obs_t = torch.as_tensor(obs_flat, device=self.device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            q_vals = self.q_network(obs_t)
            q_vals[~mask_t] = -1e8
            actions = q_vals.argmax(dim=1)
        return actions.cpu().numpy()

    def slice(self, env_idx: int) -> "DQNSlicedPlayer":
        return DQNSlicedPlayer(self)


class DQNSlicedPlayer(SlicedPlayer):
    """Per-env proxy into DQNBatchedPlayer."""

    def __init__(self, batched: DQNBatchedPlayer):
        self.batched = batched

    def action(self, obs, mask) -> int:
        # Expand single obs to batch of 1
        cls = type(obs)
        obs_batch = cls(**{f: np.expand_dims(getattr(obs, f), 0) for f in obs._fields})
        mask_batch = np.expand_dims(mask, 0)
        return int(self.batched.batch_action(obs_batch, mask_batch, [0])[0])


# ── Trainer ─────────────────────────────────────────────────────────────────

def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQNTrainer:
    """Game-agnostic DQN trainer for VecSinglePlayerEnv."""

    BENCHMARK_ENVS = 32
    BENCHMARK_GAMES = 100

    def __init__(self, config: DQNConfig, q_network: BaseQNetwork,
                 target_network: BaseQNetwork, opponent, envs, device):
        self.config = config
        self.q_network = q_network
        self.target_network = target_network
        self.opponent = opponent
        self.envs = envs
        self.device = device

        self.target_network.load_state_dict(q_network.state_dict())
        self.optimizer = optim.Adam(q_network.parameters(), lr=config.learning_rate)

    def _tb_target(self, rb, target_network, s_idx, s_rewards, s_next_obs, s_next_masks, s_dones):
        """Sample-time n-step tree-backup target (Sutton & Barto 7.5, greedy target):
        G = sum_{k<m} g^k R_{k+1} + g^m max_a Q(s_m), where m = first intermediate
        action that is NOT greedy under the CURRENT target net (else m=n). Walks the
        raw 1-step window; same-env next step lives at stride num_envs. The g^k power
        is exact (g=1 here). (A <0.01% fraction of windows may straddle the buffer
        write seam and cut early — negligible, unbiased noise.)"""
        cfg = self.config
        ne, bs, n, g, dev = rb.num_envs, rb.buffer_size, cfg.n_step, cfg.gamma, self.device
        td = s_rewards.clone()                         # R_1 (anchor reward, k=0)
        acc = s_rewards.clone()                        # sum_{j<=k} g^j R_{j+1}
        alive = (s_dones == 0)                         # trace still open past anchor
        boot_obs, boot_mask = s_next_obs, s_next_masks # bootstrap state if window runs full
        idx = s_idx
        for k in range(1, n):
            idx = (idx + ne) % bs
            obs_k = torch.as_tensor(rb.obs[idx], device=dev)
            mask_k = torch.as_tensor(rb.masks[idx], dtype=torch.bool, device=dev)
            act_k = torch.as_tensor(rb.actions[idx], dtype=torch.long, device=dev)
            rew_k = torch.as_tensor(rb.rewards[idx], device=dev)
            done_k = torch.as_tensor(rb.dones[idx], device=dev)
            q_k = target_network(obs_k); q_k[~mask_k] = -1e8
            cut = alive & (act_k != q_k.argmax(1))     # non-greedy under current Q -> cut
            td = torch.where(cut, acc + (g ** k) * q_k.max(1).values, td)
            alive = alive & ~cut
            acc = acc + alive.float() * (g ** k) * rew_k
            term = alive & (done_k != 0)               # episode ended inside the window
            td = torch.where(term, acc, td)
            alive = alive & ~term
            nobs_k = torch.as_tensor(rb.next_obs[idx], device=dev)
            nmask_k = torch.as_tensor(rb.next_masks[idx], dtype=torch.bool, device=dev)
            boot_obs = torch.where(alive.unsqueeze(1), nobs_k, boot_obs)
            boot_mask = torch.where(alive.unsqueeze(1), nmask_k, boot_mask)
        q_f = target_network(boot_obs); q_f[~boot_mask] = -1e8
        td = torch.where(alive, acc + (g ** n) * q_f.max(1).values, td)
        return td

    def train(self):
        config = self.config
        device = self.device
        q_network = self.q_network
        target_network = self.target_network
        envs = self.envs
        N = config.num_envs

        # Init replay buffer
        obs, masks = envs.reset()
        o_dim = _obs_dim(obs)
        n_act = q_network.num_actions
        rb = ReplayBuffer(config.buffer_size, o_dim, n_act, device,
                          n_step=config.n_step, gamma=config.gamma, num_envs=N,
                          prioritized=config.prioritized, alpha=config.per_alpha,
                          tree_backup=config.tree_backup)
        gamma_n = config.gamma ** config.n_step   # Rainbow: n-step bootstrap discount
        # Rainbow: C51 support (only used when config.distributional)
        support = torch.linspace(config.v_min, config.v_max, config.n_atoms, device=device)
        delta_z = (config.v_max - config.v_min) / (config.n_atoms - 1)

        obs_flat = _flatten_obs_batch(obs)
        start_time = time.time()
        global_step = 0
        num_updates = 0
        ep_returns = []
        ep_scores = []

        try:
            import wandb
            has_wandb = wandb.run is not None
        except ImportError:
            has_wandb = False

        while global_step < config.total_timesteps:
            # ── ε-greedy action selection ──
            epsilon = linear_schedule(
                config.start_e, config.end_e,
                config.exploration_fraction * config.total_timesteps, global_step)

            # Random actions for exploration
            explore = np.random.random(N) < epsilon
            # Greedy actions from Q-network
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_flat, device=device)
                q_values = q_network(obs_t)  # (N, n_act)
                # Mask illegal actions
                mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
                q_values[~mask_t] = -1e8
                greedy_actions = q_values.argmax(dim=1).cpu().numpy()

            # Sample random legal actions for exploring envs
            actions = greedy_actions.copy()
            for i in range(N):
                if explore[i]:
                    legal = np.where(masks[i])[0]
                    actions[i] = np.random.choice(legal)

            # ── Step environments ──
            next_obs, next_masks, rewards, terminated, truncated, infos = envs.step(actions)
            next_obs_flat = _flatten_obs_batch(next_obs)

            # ── Log episode completions ──
            for i in range(N):
                if terminated[i]:
                    r = float(rewards[i])
                    ep_returns.append(r)
                    scores = infos[i].get("final_scores")
                    seat = infos[i].get("agent_seat", 0)
                    if scores is not None:
                        ep_scores.append(float(np.asarray(scores)[seat]))

            # ── Store transitions ──
            rb.add_batch(obs_flat, next_obs_flat, masks, next_masks,
                         actions, rewards, terminated.astype(np.float32), explore=explore)

            obs_flat = next_obs_flat
            obs = next_obs
            masks = next_masks
            global_step += N

            # ── Training ──
            if global_step >= config.learning_starts and global_step % config.train_frequency == 0:
                # Rainbow/PER: anneal the IS exponent beta 0.4 → 1 over training
                beta = config.per_beta + (1 - config.per_beta) * (global_step / config.total_timesteps)
                for _ in range(config.grad_steps_per_train):
                    s_obs, s_next_obs, s_masks, s_next_masks, s_actions, s_rewards, s_dones, \
                        s_weights, s_idx = rb.sample(config.batch_size, beta)

                    if config.distributional:
                        # ── C51 distributional target (n-step + optional double) ──
                        with torch.no_grad():
                            next_dist = q_network.dist(s_next_obs)          # (B, A, atoms)
                            next_q = (next_dist * support).sum(-1)
                            next_q[~s_next_masks] = -1e8
                            if config.double_dqn:
                                next_act = next_q.argmax(1)
                            else:
                                tgt_q = (target_network.dist(s_next_obs) * support).sum(-1)
                                tgt_q[~s_next_masks] = -1e8
                                next_act = tgt_q.argmax(1)
                            next_pmf = target_network.dist(s_next_obs)[torch.arange(config.batch_size), next_act]
                            tz = (s_rewards.unsqueeze(1)
                                  + gamma_n * support.unsqueeze(0) * (1 - s_dones.unsqueeze(1))).clamp(config.v_min, config.v_max)
                            b = (tz - config.v_min) / delta_z
                            lo, up = b.floor().clamp(0, config.n_atoms - 1), b.ceil().clamp(0, config.n_atoms - 1)
                            d_lo = (up + (lo == b).float() - b) * next_pmf
                            d_up = (b - lo) * next_pmf
                            target_pmf = torch.zeros_like(next_pmf)
                            for i in range(config.batch_size):
                                target_pmf[i].index_add_(0, lo[i].long(), d_lo[i])
                                target_pmf[i].index_add_(0, up[i].long(), d_up[i])
                        pred_pmf = q_network.dist(s_obs)[torch.arange(config.batch_size), s_actions]
                        log_pred = torch.log(pred_pmf.clamp(1e-5, 1 - 1e-5))
                        per_sample = -(target_pmf * log_pred).sum(1)            # cross-entropy
                        current_q = (pred_pmf * support).sum(-1)               # for logging / priorities
                    else:
                        # ── Scalar TD target (n-step + optional double) ──
                        with torch.no_grad():
                            if config.tree_backup:
                                td_target = self._tb_target(rb, target_network, s_idx,
                                    s_rewards, s_next_obs, s_next_masks, s_dones)
                            else:
                                target_q = target_network(s_next_obs)
                                target_q[~s_next_masks] = -1e8
                                if config.double_dqn:
                                    online_next = q_network(s_next_obs)
                                    online_next[~s_next_masks] = -1e8
                                    next_act = online_next.argmax(dim=1)
                                    target_max = target_q.gather(1, next_act.unsqueeze(1)).squeeze(1)
                                else:
                                    target_max = target_q.max(dim=1).values
                                td_target = s_rewards + gamma_n * target_max * (1 - s_dones)
                        current_q = q_network(s_obs).gather(1, s_actions.unsqueeze(1)).squeeze(1)
                        per_sample = (F.smooth_l1_loss(current_q, td_target, reduction="none") if config.huber_loss
                                      else F.mse_loss(current_q, td_target, reduction="none"))

                    loss = (s_weights * per_sample).mean()   # PER IS-weights (=1 when off)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_network.parameters(), config.max_grad_norm)
                    self.optimizer.step()
                    num_updates += 1

                    if config.prioritized:
                        rb.update_priorities(s_idx, per_sample.detach().cpu().numpy())

                # ── Update target network ──
                if num_updates % config.target_network_frequency == 0:
                    for tp, qp in zip(target_network.parameters(), q_network.parameters()):
                        tp.data.copy_(config.tau * qp.data + (1 - config.tau) * tp.data)

            # ── Logging ──
            if global_step % config.log_interval < N:
                elapsed = time.time() - start_time
                sps = int(global_step / elapsed) if elapsed > 0 else 0
                mean_ret = np.mean(ep_returns[-100:]) if ep_returns else 0
                mean_score = np.mean(ep_scores[-100:]) if ep_scores else 0
                buf_pct = rb.size() / config.buffer_size * 100
                print(f"Step: {global_step}  SPS: {sps}  ε={epsilon:.3f}  "
                      f"ret={mean_ret:.3f}  score={mean_score:.1f}  "
                      f"buf={buf_pct:.0f}%  updates={num_updates}")

                if has_wandb:
                    log = {
                        "global_step": global_step,
                        "charts/SPS": sps,
                        "charts/epsilon": epsilon,
                        "charts/buffer_size": rb.size(),
                        "charts/mean_return": mean_ret,
                        "charts/mean_score": mean_score,
                    }
                    if global_step >= config.learning_starts:
                        log["losses/td_loss"] = loss.item()
                        log["losses/q_values"] = current_q.mean().item()
                    wandb.log(log)

            # ── Save model ──
            if global_step % config.save_interval < N:
                os.makedirs(config.model_dir, exist_ok=True)
                path = os.path.join(config.model_dir, f"{global_step}.pt")
                torch.save(q_network.state_dict(), path)

            # ── Benchmark ──
            if global_step % config.log_interval < N:
                self.benchmark(global_step)

        # Final save
        os.makedirs(config.model_dir, exist_ok=True)
        torch.save(q_network.state_dict(), os.path.join(config.model_dir, "final.pt"))

    def benchmark(self, global_step: int):
        """Override in subclass for game-specific benchmarks."""
        pass
