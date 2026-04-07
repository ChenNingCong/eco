#!/usr/bin/env python3
"""
Neural Fictitious Self-Play (NFSP) for Lost Cities.

Two-headed agent:
  - Best Response (RL): DQN that learns to beat the opponent's average policy
  - Average Policy (SL): Supervised network that imitates the RL head's action history

Algorithm:
  1. Each episode, set policy σ = ε-greedy(Q) with prob η, else Π (average policy)
  2. Execute σ, store transitions in M_RL; if using best-response, also store (s,a) in M_SL
  3. Train Q on M_RL (standard DQN loss)
  4. Train Π on M_SL (cross-entropy loss)

Usage:
    python -m game.lc.train_nfsp --total-timesteps 10000000 --track
"""
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

from abstract import VecSinglePlayerEnv, RandomPlayer, key_from_seed
from abstract.dqn import (
    ReplayBuffer, DQNBatchedPlayer, _flatten_obs_batch, _obs_dim, linear_schedule,
)
from abstract.ppo_lstm import layer_init
from game.lc import LCEnvFactory
from game.lc.dqn_agent import LCQNetwork
from game.lc.engine import float_dim, NUM_ACTIONS


# ── Config ─────────────────────────────────────────────────────────────────

@dataclass
class NFSPConfig:
    """NFSP hyperparameters for Lost Cities."""
    exp_name: str = "lc_nfsp"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ppo-eco"
    wandb_entity: Optional[str] = None

    # ── Training scale ──
    total_timesteps: int = 20_000_000
    num_envs: int = 128

    # ── DQN (Best Response) ──
    rl_lr: float = 1e-4
    rl_buffer_size: int = 1_000_000
    rl_batch_size: int = 256
    rl_gamma: float = 1.0
    rl_start_e: float = 1.0
    rl_end_e: float = 0.05
    rl_exploration_fraction: float = 0.10
    rl_learning_starts: int = 10_000
    rl_train_frequency: int = 4
    rl_target_network_frequency: int = 1000
    rl_tau: float = 1.0
    rl_max_grad_norm: float = 10.0

    # ── SL (Average Policy) ──
    sl_lr: float = 1e-4
    sl_buffer_size: int = 2_000_000
    """reservoir buffer size for SL data"""
    sl_batch_size: int = 256
    sl_train_frequency: int = 4
    """train SL every N env steps"""
    sl_max_grad_norm: float = 10.0

    # ── NFSP-specific ──
    eta: float = 0.1
    """anticipatory parameter: prob of using best-response (RL) vs average (SL)"""

    # ── Architecture ──
    hidden_dim: int = 256

    # ── Game ──
    new_color_penalty: int = 20
    max_lanes: int = 5
    max_discard_draws: int = 50
    """max total discard draws before masking; needed to prevent infinite discard loops"""
    dense_reward: bool = True
    """use per-step dense reward for DQN RL head"""

    # ── Logging/saving ──
    model_dir: str = "model/lc_nfsp"
    log_interval: int = 81_920
    save_interval: int = 819_200


# ── Average Policy Network ─────────────────────────────────────────────────

class AveragePolicyNetwork(nn.Module):
    """Policy network π(a|s) for the SL head. Same architecture as Q-network
    but outputs log-probabilities instead of Q-values."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        obs_dim = float_dim()
        H = hidden_dim

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, H)), nn.LayerNorm(H), nn.ReLU(),
            layer_init(nn.Linear(H, H)),       nn.LayerNorm(H), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(H, H)), nn.LayerNorm(H), nn.ReLU(),
            layer_init(nn.Linear(H, H)), nn.LayerNorm(H), nn.ReLU(),
        )
        self.head = layer_init(nn.Linear(H, NUM_ACTIONS), std=0.01)

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, num_actions)."""
        enc = self.encoder(obs_flat[:, :float_dim()])
        return self.head(self.trunk(enc))

    def get_action(self, obs_flat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sample action from masked policy."""
        logits = self.forward(obs_flat)
        logits[~mask] = -1e8
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(1)

    def get_greedy_action(self, obs_flat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Greedy action from masked policy (for evaluation)."""
        logits = self.forward(obs_flat)
        logits[~mask] = -1e8
        return logits.argmax(dim=1)


# ── Reservoir Sampling Buffer ──────────────────────────────────────────────

class ReservoirBuffer:
    """Reservoir sampling buffer for SL data: stores (obs_flat, action) pairs.
    Ensures uniform sampling over entire history regardless of buffer size."""

    def __init__(self, capacity: int, obs_dim: int, device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.count = 0  # total items ever added
        self.size = 0   # current number of items stored

    def add(self, obs_flat: np.ndarray, action: int):
        """Add single (obs, action) pair with reservoir sampling."""
        if self.size < self.capacity:
            self.obs[self.size] = obs_flat
            self.actions[self.size] = action
            self.size += 1
        else:
            idx = np.random.randint(0, self.count + 1)
            if idx < self.capacity:
                self.obs[idx] = obs_flat
                self.actions[idx] = action
        self.count += 1

    def add_batch(self, obs_flat: np.ndarray, actions: np.ndarray):
        """Add batch of (obs, action) pairs."""
        for i in range(len(actions)):
            self.add(obs_flat[i], actions[i])

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], dtype=torch.long, device=self.device),
        )


# ── NFSP Batched Player (for opponent) ─────────────────────────────────────

class NFSPBatchedPlayer:
    """NFSP opponent: uses average policy (SL head) for opponent actions.
    This is what the RL head trains against."""

    def __init__(self, sl_network: AveragePolicyNetwork, device, num_envs: int):
        self.sl_network = sl_network
        self.device = device
        self.num_envs = num_envs

    def reset(self, env_indices=None):
        pass

    def batch_action(self, obs_batch, mask_batch: np.ndarray, idxs: list) -> np.ndarray:
        obs_flat = _flatten_obs_batch(obs_batch)
        obs_t = torch.as_tensor(obs_flat, device=self.device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            actions = self.sl_network.get_action(obs_t, mask_t)
        return actions.cpu().numpy()

    def slice(self, env_idx: int):
        return NFSPSlicedPlayer(self)


class NFSPSlicedPlayer:
    def __init__(self, batched: NFSPBatchedPlayer):
        self.batched = batched

    def action(self, obs, mask) -> int:
        cls = type(obs)
        obs_batch = cls(**{f: np.expand_dims(getattr(obs, f), 0) for f in obs._fields})
        mask_batch = np.expand_dims(mask, 0)
        return int(self.batched.batch_action(obs_batch, mask_batch, [0])[0])


# ── Benchmark ──────────────────────────────────────────────────────────────

def run_benchmark(q_net, sl_net, device, factory, seed, n_games=100, n_envs=32):
    """Benchmark both RL and SL heads vs random."""
    results = {}
    for label, get_action_fn in [
        ("rl_vs_random", lambda obs_t, mask_t: (
            q_net(obs_t).masked_fill(~mask_t, -1e8).argmax(dim=1)
        )),
        ("sl_vs_random", lambda obs_t, mask_t: sl_net.get_greedy_action(obs_t, mask_t)),
    ]:
        opponent = RandomPlayer()
        key = key_from_seed(seed)
        envs = VecSinglePlayerEnv(num_envs=n_envs, opponent=opponent,
                                  env_factory=factory, key=key)
        obs, masks = envs.reset()
        wins, losses, total = 0, 0, 0
        scores = []

        while total < n_games:
            obs_flat = _flatten_obs_batch(obs)
            obs_t = torch.as_tensor(obs_flat, device=device)
            mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                actions = get_action_fn(obs_t, mask_t).cpu().numpy()
            obs, masks, rewards, terminated, truncated, infos = envs.step(actions)
            for i, t in enumerate(terminated):
                if t:
                    total += 1
                    s = infos[i].get("final_scores")
                    seat = infos[i].get("agent_seat", 0)
                    if s is not None:
                        sc = float(np.asarray(s)[seat])
                        opp_sc = float(np.asarray(s)[1 - seat])
                        scores.append(sc)
                        if sc > opp_sc:
                            wins += 1
                        elif sc < opp_sc:
                            losses += 1
        envs.close()
        wr = wins / total if total else 0
        ms = np.mean(scores) if scores else 0
        results[f"{label}/win_rate"] = wr
        results[f"{label}/mean_score"] = ms
        results[f"{label}/wins"] = wins
        results[f"{label}/losses"] = losses
        print(f"  {label}: {wins}W/{losses}L/{total-wins-losses}D  score={ms:.1f}")
    return results


# ── Main Training Loop ─────────────────────────────────────────────────────

def train(cfg: NFSPConfig):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    # Networks
    q_network = LCQNetwork(hidden_dim=cfg.hidden_dim).to(device)
    target_network = LCQNetwork(hidden_dim=cfg.hidden_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    sl_network = AveragePolicyNetwork(hidden_dim=cfg.hidden_dim).to(device)

    print(f"Q-network params: {sum(p.numel() for p in q_network.parameters()):,}")
    print(f"SL-network params: {sum(p.numel() for p in sl_network.parameters()):,}")

    rl_optimizer = torch.optim.Adam(q_network.parameters(), lr=cfg.rl_lr)
    sl_optimizer = torch.optim.Adam(sl_network.parameters(), lr=cfg.sl_lr)

    # Opponent = average policy (SL head)
    opponent = NFSPBatchedPlayer(sl_network, device, cfg.num_envs)

    # Environment
    factory = LCEnvFactory(
        new_color_penalty=cfg.new_color_penalty,
        max_lanes=cfg.max_lanes,
        max_discard_draws=cfg.max_discard_draws,
        dense_reward=cfg.dense_reward,
    )
    key = key_from_seed(cfg.seed)
    envs = VecSinglePlayerEnv(num_envs=cfg.num_envs, opponent=opponent,
                              env_factory=factory, key=key)

    # Buffers
    obs, masks = envs.reset()
    o_dim = _obs_dim(obs)
    n_act = NUM_ACTIONS
    rl_buffer = ReplayBuffer(cfg.rl_buffer_size, o_dim, n_act, device)
    sl_buffer = ReservoirBuffer(cfg.sl_buffer_size, o_dim, device)

    obs_flat = _flatten_obs_batch(obs)

    # Per-env policy mode: True = best response (RL), False = average (SL)
    # Resampled each episode
    use_rl = np.random.random(cfg.num_envs) < cfg.eta

    # wandb
    try:
        import wandb
        has_wandb = wandb.run is not None
    except ImportError:
        has_wandb = False

    start_time = time.time()
    global_step = 0
    rl_updates = 0
    sl_updates = 0
    ep_scores = []
    rl_transitions = 0
    sl_transitions = 0

    bench_factory = LCEnvFactory(new_color_penalty=cfg.new_color_penalty,
                                 max_lanes=cfg.max_lanes,
                                 max_discard_draws=cfg.max_discard_draws)

    while global_step < cfg.total_timesteps:
        N = cfg.num_envs

        # ── Action selection ──
        # RL envs: ε-greedy(Q)
        # SL envs: sample from Π
        epsilon = linear_schedule(
            cfg.rl_start_e, cfg.rl_end_e,
            cfg.rl_exploration_fraction * cfg.total_timesteps, global_step)

        obs_t = torch.as_tensor(obs_flat, device=device)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)

        with torch.no_grad():
            # Q-values for RL envs
            q_values = q_network(obs_t)
            q_values_masked = q_values.clone()
            q_values_masked[~mask_t] = -1e8
            greedy_actions = q_values_masked.argmax(dim=1).cpu().numpy()

            # SL actions
            sl_actions = sl_network.get_action(obs_t, mask_t).cpu().numpy()

        # Combine: RL envs use ε-greedy, SL envs use Π
        actions = np.where(use_rl, greedy_actions, sl_actions)

        # ε-exploration for RL envs
        explore = np.random.random(N) < epsilon
        for i in range(N):
            if use_rl[i] and explore[i]:
                legal = np.where(masks[i])[0]
                actions[i] = np.random.choice(legal)

        # ── Step environments ──
        next_obs, next_masks, rewards, terminated, truncated, infos = envs.step(actions)
        next_obs_flat = _flatten_obs_batch(next_obs)

        # ── Store in buffers ──
        # M_RL: all transitions (both RL and SL envs contribute)
        rl_buffer.add_batch(obs_flat, next_obs_flat, masks, next_masks,
                            actions, rewards, terminated.astype(np.float32))

        # M_SL: only transitions from RL (best-response) envs
        rl_mask = use_rl
        if rl_mask.any():
            rl_idxs = np.where(rl_mask)[0]
            sl_buffer.add_batch(obs_flat[rl_idxs], actions[rl_idxs])
            sl_transitions += len(rl_idxs)

        rl_transitions += N

        # ── Log episode completions & resample policy mode ──
        for i in range(N):
            if terminated[i]:
                scores = infos[i].get("final_scores")
                seat = infos[i].get("agent_seat", 0)
                if scores is not None:
                    ep_scores.append(float(np.asarray(scores)[seat]))
                # Resample policy mode for new episode
                use_rl[i] = np.random.random() < cfg.eta

        obs_flat = next_obs_flat
        obs = next_obs
        masks = next_masks
        global_step += N

        # ── Train RL (DQN) ──
        if global_step >= cfg.rl_learning_starts and global_step % cfg.rl_train_frequency == 0:
            s_obs, s_next_obs, s_masks, s_next_masks, s_actions, s_rewards, s_dones = \
                rl_buffer.sample(cfg.rl_batch_size)

            with torch.no_grad():
                target_q = target_network(s_next_obs)
                target_q[~s_next_masks] = -1e8
                target_max = target_q.max(dim=1).values
                td_target = s_rewards + cfg.rl_gamma * target_max * (1 - s_dones)

            current_q = q_network(s_obs).gather(1, s_actions.unsqueeze(1)).squeeze(1)
            rl_loss = F.mse_loss(td_target, current_q)

            rl_optimizer.zero_grad()
            rl_loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), cfg.rl_max_grad_norm)
            rl_optimizer.step()
            rl_updates += 1

            # Update target network
            if rl_updates % cfg.rl_target_network_frequency == 0:
                for tp, qp in zip(target_network.parameters(), q_network.parameters()):
                    tp.data.copy_(cfg.rl_tau * qp.data + (1 - cfg.rl_tau) * tp.data)

        # ── Train SL (Average Policy) ──
        if sl_buffer.size >= cfg.sl_batch_size and global_step % cfg.sl_train_frequency == 0:
            s_obs, s_actions = sl_buffer.sample(cfg.sl_batch_size)
            logits = sl_network(s_obs)
            sl_loss = F.cross_entropy(logits, s_actions)

            sl_optimizer.zero_grad()
            sl_loss.backward()
            nn.utils.clip_grad_norm_(sl_network.parameters(), cfg.sl_max_grad_norm)
            sl_optimizer.step()
            sl_updates += 1

        # ── Logging ──
        if global_step % cfg.log_interval < N:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            mean_score = np.mean(ep_scores[-100:]) if ep_scores else 0
            rl_buf_pct = rl_buffer.size() / cfg.rl_buffer_size * 100
            sl_buf_pct = sl_buffer.size / cfg.sl_buffer_size * 100
            print(f"Step: {global_step}  SPS: {sps}  ε={epsilon:.3f}  "
                  f"score={mean_score:.1f}  "
                  f"rl_buf={rl_buf_pct:.0f}%  sl_buf={sl_buf_pct:.0f}%  "
                  f"rl_upd={rl_updates}  sl_upd={sl_updates}")

            if has_wandb:
                log = {
                    "global_step": global_step,
                    "charts/SPS": sps,
                    "charts/epsilon": epsilon,
                    "charts/mean_score": mean_score,
                    "charts/rl_buffer_pct": rl_buf_pct,
                    "charts/sl_buffer_pct": sl_buf_pct,
                    "charts/rl_updates": rl_updates,
                    "charts/sl_updates": sl_updates,
                    "charts/sl_transitions": sl_transitions,
                }
                if rl_updates > 0:
                    log["losses/rl_loss"] = rl_loss.item()
                    log["losses/q_values"] = current_q.mean().item()
                if sl_updates > 0:
                    log["losses/sl_loss"] = sl_loss.item()
                wandb.log(log)

            # Benchmark
            q_network.eval()
            sl_network.eval()
            bench_log = run_benchmark(q_network, sl_network, device,
                                      bench_factory, cfg.seed + 10000)
            q_network.train()
            sl_network.train()
            if has_wandb:
                wandb.log({f"benchmark/{k}": v for k, v in bench_log.items()})

        # ── Save ──
        if global_step % cfg.save_interval < N:
            os.makedirs(cfg.model_dir, exist_ok=True)
            torch.save({
                "q_network": q_network.state_dict(),
                "target_network": target_network.state_dict(),
                "sl_network": sl_network.state_dict(),
                "global_step": global_step,
            }, os.path.join(cfg.model_dir, f"nfsp_{global_step}.pt"))

    # Final save
    os.makedirs(cfg.model_dir, exist_ok=True)
    torch.save({
        "q_network": q_network.state_dict(),
        "target_network": target_network.state_dict(),
        "sl_network": sl_network.state_dict(),
        "global_step": global_step,
    }, os.path.join(cfg.model_dir, "nfsp_final.pt"))
    # Also save just the Q-network for bench_agents compatibility
    torch.save(q_network.state_dict(), os.path.join(cfg.model_dir, "q_network.pt"))

    envs.close()
    print("Training complete.")


def main():
    cfg = tyro.cli(NFSPConfig)
    import wandb

    run_name = f"lc__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    if cfg.track:
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            config=vars(cfg),
            name=run_name,
            save_code=True,
        )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    train(cfg)


if __name__ == "__main__":
    main()
