# abstract/ppo_lstm.py — Game-agnostic PPO+LSTM training.
#   - LSTMState pytree for multi-layer LSTM hidden states (h, c)
#   - BaseAgent: abstract actor-critic interface
#   - LSTMBatchedPlayer/LSTMSlicedPlayer: LSTM opponent state management
#   - PPOLSTMTrainer: game-agnostic training loop
import os
import time
from dataclasses import dataclass
from typing import Optional, NamedTuple
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils._pytree import tree_map

from .player import BasePlayer, SlicedPlayer

# ── PyTree helpers ───────────────────────────────────────────────────────────

def _obs_dtype(x: np.ndarray):
    """Return the torch dtype matching a PyTreeObs leaf by inspecting its numpy dtype."""
    return torch.long if np.issubdtype(x.dtype, np.integer) else torch.float32

def obs_to_tensor(obs, device):
    """numpy PyTreeObs → tensor PyTreeObs, preserving dtypes."""
    return tree_map(
        lambda x: torch.as_tensor(x, dtype=_obs_dtype(x), device=device), obs
    )

def obs_unsqueeze(obs):
    """Add batch dim to a single obs pytree (for SlicedPlayer)."""
    return tree_map(lambda x: np.expand_dims(x, 0), obs)

def alloc_obs_buffer(prototype, T: int, N: int, device):
    """Allocate a (T, N, *leaf_shape) storage buffer matching prototype dtypes."""
    return tree_map(
        lambda x: torch.zeros((T, N, *x.shape[1:]), dtype=_obs_dtype(x), device=device),
        prototype,
    )

# ── LSTMState: pytree for multi-layer LSTM hidden states ────────────────────

class LSTMState(NamedTuple):
    """Multi-layer LSTM hidden state: h and c each (num_layers, batch, hidden)."""
    h: torch.Tensor
    c: torch.Tensor

def make_lstm_state(num_layers: int, batch_size: int, hidden_size: int, device) -> LSTMState:
    """Create zero-init LSTM state."""
    return LSTMState(
        h=torch.zeros(num_layers, batch_size, hidden_size, device=device),
        c=torch.zeros(num_layers, batch_size, hidden_size, device=device),
    )

# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    """Game-agnostic PPO+LSTM hyperparameters."""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ppo"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    gae_lambda: float = 1.0
    """the lambda for the general advantage estimation."""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    ent_coef_end: float = 0.0
    """final entropy coefficient (for annealing). 0 = no annealing (use ent_coef throughout)."""
    ent_anneal_steps: int = 0
    """number of steps over which to anneal entropy from ent_coef to ent_coef_end. 0 = no annealing."""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.01
    """the target KL divergence threshold"""
    lstm_hidden: int = 128
    """LSTM hidden size."""
    model_dir: str = "model"
    """directory to save/load model checkpoints."""
    log_interval: int = 81_920
    """the interval (in samples) at which to run benchmark (~20 iterations)"""
    save_interval: int = 81_920
    """the interval (in samples) at which to save the model (~20 iterations)"""

    # computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

# ── NN helpers ───────────────────────────────────────────────────────────────

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks: Optional[torch.BoolTensor] = None):
        self.masks = masks
        if masks is None:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            _masks = masks.to(probs.device if probs is not None else logits.device)
            self.masks = _masks
            assert logits is not None
            logits = torch.where(_masks, logits, torch.tensor(-1e8))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if self.masks is None:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, 0)
        return -p_log_p.sum(-1)

# ── Abstract agent interface ─────────────────────────────────────────────────

class BaseAgent(nn.Module, ABC):
    """
    Abstract actor-critic with LSTM. Pure/stateless: hidden state passed in/out.

    Subclasses must set lstm_hidden, lstm_layers, and implement the three methods.
    """
    lstm_hidden: int
    lstm_layers: int

    @abstractmethod
    def get_states(self, obs, lstm_state: LSTMState, done: torch.Tensor):
        """Encode obs + LSTM. Returns (hidden, new_lstm_state)."""
        ...

    @abstractmethod
    def get_value(self, obs, lstm_state: LSTMState, done: torch.Tensor):
        """Returns value estimate (B, 1)."""
        ...

    @abstractmethod
    def get_action_and_value(self, obs, action_mask, lstm_state: LSTMState,
                             done: torch.Tensor, action=None):
        """Returns (action, log_prob, entropy, value, new_lstm_state)."""
        ...

    @property
    @abstractmethod
    def num_actions(self) -> int:
        ...

# ── LSTMBatchedPlayer / LSTMSlicedPlayer ────────────────────────────────────

class LSTMBatchedPlayer(BasePlayer):
    """
    Manages batched LSTM hidden states for N envs.
    The NN agent is stateless (pure); this class holds the recurrent state.
    """
    def __init__(self, agent: BaseAgent, device, num_envs: int):
        self.agent = agent
        self.device = device
        self.num_envs = num_envs
        self.lstm_state = make_lstm_state(
            agent.lstm_layers, num_envs, agent.lstm_hidden, device,
        )

    def reset(self, env_indices=None):
        """Reset LSTM state for given env indices (or all if None)."""
        if env_indices is None:
            self.lstm_state = make_lstm_state(
                self.agent.lstm_layers, self.num_envs, self.agent.lstm_hidden, self.device,
            )
        else:
            for idx in env_indices:
                self.lstm_state.h[:, idx] = 0
                self.lstm_state.c[:, idx] = 0

    def batch_action(self, obs_batch, mask_batch: np.ndarray,
                     idxs: list) -> np.ndarray:
        """Batched action for a subset of envs identified by idxs."""
        obs_t = obs_to_tensor(obs_batch, self.device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=self.device)
        done_t = torch.zeros(len(idxs), device=self.device)
        idx_t = torch.tensor(idxs, dtype=torch.long, device=self.device)
        sub_state = LSTMState(
            h=self.lstm_state.h[:, idx_t].contiguous(),
            c=self.lstm_state.c[:, idx_t].contiguous(),
        )
        with torch.no_grad():
            acts, _, _, _, new_state = self.agent.get_action_and_value(
                obs_t, mask_t, sub_state, done_t
            )
        self.lstm_state.h[:, idx_t] = new_state.h
        self.lstm_state.c[:, idx_t] = new_state.c
        return acts.cpu().numpy()

    def slice(self, env_idx: int) -> "LSTMSlicedPlayer":
        return LSTMSlicedPlayer(self, env_idx)


class LSTMSlicedPlayer(SlicedPlayer):
    """
    Per-env proxy into a LSTMBatchedPlayer. Uses abstract SlicedPlayer interface.
    """
    def __init__(self, batched: LSTMBatchedPlayer, env_idx: int):
        self.batched = batched
        self.env_idx = env_idx

    def action(self, obs, mask) -> int:
        i = self.env_idx
        player = self.batched
        obs_t = obs_to_tensor(obs_unsqueeze(obs), player.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=player.device).unsqueeze(0)
        done_t = torch.zeros(1, device=player.device)
        single_state = LSTMState(
            h=player.lstm_state.h[:, i:i+1].contiguous(),
            c=player.lstm_state.c[:, i:i+1].contiguous(),
        )
        with torch.no_grad():
            action, _, _, _, new_state = player.agent.get_action_and_value(
                obs_t, mask_t, single_state, done_t
            )
        player.lstm_state.h[:, i] = new_state.h[:, 0]
        player.lstm_state.c[:, i] = new_state.c[:, 0]
        return int(action.item())

# ── PPOLSTMTrainer ───────────────────────────────────────────────────────────

class PPOLSTMTrainer:
    """
    Game-agnostic PPO+LSTM trainer.

    Parameters
    ----------
    config      : PPOConfig
    agent       : BaseAgent on device
    opponent    : BasePlayer (for LSTM state reset on done)
    envs        : VecSinglePlayerEnv (or anything with reset/step/close)
    device      : torch device
    """

    def __init__(self, config: PPOConfig, agent: BaseAgent, opponent: BasePlayer,
                 envs, device):
        self.cfg = config
        self.agent = agent
        self.opponent = opponent
        self.envs = envs
        self.device = device

        # Compute derived config
        self.cfg.batch_size = self.cfg.num_envs * self.cfg.num_steps
        self.cfg.minibatch_size = self.cfg.batch_size // self.cfg.num_minibatches
        self.cfg.num_iterations = self.cfg.total_timesteps // self.cfg.batch_size

    def benchmark(self, global_step: int):
        """Override in subclass to run periodic benchmarks (e.g. vs random agent).
        Called every log_interval steps during training."""
        pass

    def train(self):
        cfg = self.cfg
        agent = self.agent
        opponent = self.opponent
        envs = self.envs
        device = self.device

        try:
            import wandb
        except ImportError:
            wandb = None

        optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

        # Allocate rollout buffers
        _proto_obs, _proto_masks = envs.reset()
        obs          = alloc_obs_buffer(_proto_obs, cfg.num_steps, cfg.num_envs, device)
        actions      = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long, device=device)
        logprobs     = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
        rewards      = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
        dones        = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
        values       = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
        action_masks = torch.zeros((cfg.num_steps, cfg.num_envs, agent.num_actions), dtype=torch.bool, device=device)

        next_obs  = obs_to_tensor(_proto_obs, device)
        next_masks = torch.as_tensor(_proto_masks, dtype=torch.bool, device=device)
        next_done = torch.zeros(cfg.num_envs, device=device)
        next_lstm_state = make_lstm_state(agent.lstm_layers, cfg.num_envs, agent.lstm_hidden, device)

        global_step = 0
        last_log_step = 0
        last_save_step = 0
        start_time = time.time()
        alpha = cfg.learning_rate

        for iteration in range(1, cfg.num_iterations + 1):
            initial_lstm_state = LSTMState(
                h=next_lstm_state.h.clone(),
                c=next_lstm_state.c.clone(),
            )

            # LR annealing
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
                optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate
            if cfg.target_kl is not None:
                optimizer.param_groups[0]["lr"] = alpha

            # Entropy coefficient annealing
            if cfg.ent_coef_end > 0 and cfg.ent_anneal_steps > 0:
                steps_so_far = (iteration - 1) * cfg.batch_size
                frac = min(steps_so_far / cfg.ent_anneal_steps, 1.0)
                ent_coef_now = cfg.ent_coef + frac * (cfg.ent_coef_end - cfg.ent_coef)
            else:
                ent_coef_now = cfg.ent_coef

            # ── Rollout ──────────────────────────────────────────────────
            for step in range(cfg.num_steps):
                global_step += cfg.num_envs
                tree_map(lambda buf, val: buf.__setitem__(step, val), obs, next_obs)
                dones[step] = next_done
                action_masks[step] = next_masks

                with torch.no_grad():
                    action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                        next_obs, action_masks[step], next_lstm_state, next_done
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs_np, next_masks_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                done_indices = list(np.where(next_done)[0])
                if done_indices:
                    opponent.reset(done_indices)
                rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device).view(-1)
                next_obs   = obs_to_tensor(next_obs_np, device)
                next_masks = torch.as_tensor(next_masks_np, dtype=torch.bool, device=device)
                next_done  = torch.Tensor(next_done).to(device)

                if wandb and wandb.run is not None:
                    for i in range(len(infos)):
                        if infos[i].get("final_scores") is not None:
                            wandb.log({"charts/episodic_return": float(reward[i]), "global_step": global_step})

            # LSTM hidden state diagnostics
            lstm_h_norm = next_lstm_state.h.norm().item() / (cfg.num_envs ** 0.5)
            lstm_c_norm = next_lstm_state.c.norm().item() / (cfg.num_envs ** 0.5)

            # ── GAE ──────────────────────────────────────────────────────
            with torch.no_grad():
                next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # ── Flatten ──────────────────────────────────────────────────
            b_obs = tree_map(lambda x: x.reshape(-1, *x.shape[2:]), obs)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_dones = dones.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_action_masks = action_masks.reshape((-1, action_masks.shape[-1]))

            # ── PPO update (sequential minibatching for LSTM) ────────────
            clipfracs = []
            assert cfg.num_envs % cfg.num_minibatches == 0
            envsperbatch = cfg.num_envs // cfg.num_minibatches
            envinds = np.arange(cfg.num_envs)
            flatinds = np.arange(cfg.batch_size).reshape(cfg.num_steps, cfg.num_envs)

            for epoch in range(cfg.update_epochs):
                np.random.shuffle(envinds)
                mb_inds_list = []
                mb_envinds_list = []
                for start in range(0, cfg.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_envinds_list.append(mbenvinds)
                    mb_inds_list.append(flatinds[:, mbenvinds].ravel())

                for mb_inds, mbenvinds in zip(mb_inds_list, mb_envinds_list):
                    mb_obs = tree_map(lambda x: x[mb_inds], b_obs)
                    mb_lstm_state = LSTMState(
                        h=initial_lstm_state.h[:, mbenvinds],
                        c=initial_lstm_state.c[:, mbenvinds],
                    )
                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                        mb_obs, b_action_masks[mb_inds],
                        mb_lstm_state,
                        b_dones[mb_inds],
                        b_actions.long()[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if cfg.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef_now * entropy_loss + v_loss * cfg.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm_pre = torch.nn.utils.get_total_norm(
                        [p.grad for p in agent.parameters() if p.grad is not None], 2.0
                    ).item()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    grad_norm_post = torch.nn.utils.get_total_norm(
                        [p.grad for p in agent.parameters() if p.grad is not None], 2.0
                    ).item()
                    optimizer.step()

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

            # Adaptive LR based on KL
            if cfg.target_kl is not None:
                if approx_kl > 2.0 * cfg.target_kl:
                    alpha = max(1e-5, alpha / 1.5)
                elif approx_kl < 0.5 * cfg.target_kl:
                    alpha = min(1e-2, alpha * 1.5)

            # ── Logging ──────────────────────────────────────────────────
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if wandb and wandb.run is not None:
                log_dict = {
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "grad/total_pre_clip": grad_norm_pre,
                    "grad/total_post_clip": grad_norm_post,
                    "lstm/h_norm": lstm_h_norm,
                    "lstm/c_norm": lstm_c_norm,
                    "charts/ent_coef": ent_coef_now,
                    "global_step": global_step,
                }
                wandb.log(log_dict)

            if global_step >= last_log_step + cfg.log_interval:
                self.benchmark(global_step)
                last_log_step = global_step

            if global_step >= last_save_step + cfg.save_interval and global_step > 0:
                os.makedirs(cfg.model_dir, exist_ok=True)
                torch.save(agent.state_dict(), os.path.join(cfg.model_dir, f"ckpt_{global_step}.pkt"))
                torch.save(agent.state_dict(), os.path.join(cfg.model_dir, "latest.pkt"))
                last_save_step = global_step

            sps = int(global_step / (time.time() - start_time))
            print(f"Step: {global_step} SPS: {sps}  "
                  f"ev={explained_var:.3f}  ent={entropy_loss.item():.3f}  "
                  f"grad={grad_norm_pre:.2f}/{grad_norm_post:.2f}  "
                  f"h={lstm_h_norm:.1f} c={lstm_c_norm:.1f}  "
                  f"kl={approx_kl.item():.4f}  vl={v_loss.item():.4f}")
            if wandb and wandb.run is not None:
                wandb.log({"charts/SPS": sps, "global_step": global_step})

        envs.close()
        if wandb and wandb.run is not None:
            wandb.finish()
