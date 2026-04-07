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

    def __init__(self, buffer_size: int, obs_dim: int, num_actions: int, device):
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

    def add(self, obs, next_obs, mask, next_mask, action, reward, done):
        """Add a single transition."""
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.masks[self.pos] = mask
        self.next_masks[self.pos] = next_mask
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def add_batch(self, obs, next_obs, masks, next_masks, actions, rewards, dones):
        """Add a batch of transitions. obs/next_obs are (N, D) flat arrays."""
        N = len(actions)
        if self.pos + N <= self.buffer_size:
            sl = slice(self.pos, self.pos + N)
            self.obs[sl] = obs
            self.next_obs[sl] = next_obs
            self.masks[sl] = masks
            self.next_masks[sl] = next_masks
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.dones[sl] = dones
            self.pos += N
            if self.pos >= self.buffer_size:
                self.full = True
                self.pos = self.pos % self.buffer_size
        else:
            # Wrap around
            for i in range(N):
                self.add(obs[i], next_obs[i], masks[i], next_masks[i],
                         actions[i], rewards[i], dones[i])

    def sample(self, batch_size: int):
        upper = self.buffer_size if self.full else self.pos
        idx = np.random.randint(0, upper, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.masks[idx], dtype=torch.bool, device=self.device),
            torch.as_tensor(self.next_masks[idx], dtype=torch.bool, device=self.device),
            torch.as_tensor(self.actions[idx], dtype=torch.long, device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
        )

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
        rb = ReplayBuffer(config.buffer_size, o_dim, n_act, device)

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
                         actions, rewards, terminated.astype(np.float32))

            obs_flat = next_obs_flat
            obs = next_obs
            masks = next_masks
            global_step += N

            # ── Training ──
            if global_step >= config.learning_starts and global_step % config.train_frequency == 0:
                for _ in range(config.grad_steps_per_train):
                    s_obs, s_next_obs, s_masks, s_next_masks, s_actions, s_rewards, s_dones = \
                        rb.sample(config.batch_size)

                    with torch.no_grad():
                        target_q = target_network(s_next_obs)
                        # Mask illegal next actions
                        target_q[~s_next_masks] = -1e8
                        target_max = target_q.max(dim=1).values
                        td_target = s_rewards + config.gamma * target_max * (1 - s_dones)

                    current_q = q_network(s_obs).gather(1, s_actions.unsqueeze(1)).squeeze(1)
                    loss = F.mse_loss(td_target, current_q)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_network.parameters(), config.max_grad_norm)
                    self.optimizer.step()
                    num_updates += 1

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
