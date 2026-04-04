# eco_ppo_lstm.py — PPO+LSTM training for R-öko.
# Based on eco_ppo.py, adds LSTM recurrence (following CleanRL ppo_atari_lstm.py):
#   - LSTMState pytree for multi-layer LSTM hidden states (h, c)
#   - EcoAgent: LSTM between encoder and actor/critic (pure — state passed in/out)
#   - BatchedPlayer/SlicedPlayer: clean OOP for stateful LSTM opponents
#   - Training loop: carries LSTM state, resets on done, sequential minibatch indexing
import glob
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, NamedTuple
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils._pytree import tree_map
from eco_vec_env import VecSinglePlayerEcoEnv
from eco_obs_encoder import EcoPyTreeObs, eco_float_dim
from eco_env import NUM_ACTIONS, MAX_ECO_SCORE

# ── PyTree helpers (only 6 lines in the training loop touch obs) ─────────────

def _obs_dtype(x: np.ndarray):
    """Return the torch dtype matching a PyTreeObs leaf by inspecting its numpy dtype."""
    return torch.long if np.issubdtype(x.dtype, np.integer) else torch.float32

def obs_to_tensor(obs: EcoPyTreeObs, device) -> EcoPyTreeObs:
    """numpy PyTreeObs → tensor PyTreeObs, preserving dtypes."""
    return tree_map(
        lambda x: torch.as_tensor(x, dtype=_obs_dtype(x), device=device), obs
    )

def alloc_obs_buffer(prototype: EcoPyTreeObs, T: int, N: int, device) -> EcoPyTreeObs:
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
    return LSTMState(
        h=torch.zeros(num_layers, batch_size, hidden_size, device=device),
        c=torch.zeros(num_layers, batch_size, hidden_size, device=device),
    )
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Args:
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
    wandb_project_name: str = "ppo-eco"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Eco-v0"
    """the id of the environment"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_players: int = 2
    """number of players per game"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    reward_shaping_scale: float = 1.0
    """scaling factor for per-action intermediate rewards (0 = disabled)."""
    opponent_penalty: float = 0.5
    """penalty for opponent scoring in shaping reward: r = my_r - penalty * max(opp_r)."""
    relative_seat: bool = True
    """use relative seat encoding (constant 0) instead of absolute seat index."""
    score_shortcut: bool = False
    """add a shallow score-to-value shortcut in the critic (bypasses deep trunk)."""
    model_dir: str = "model"
    """directory to save/load model checkpoints."""
    opponent_mode: Literal["self_play", "random"] = "self_play"
    """opponent policy used during training:
      self_play  — opponent uses the current agent weights (default)
      random     — opponent plays uniformly at random"""
    gae_lambda: float = 1.0
    """the lambda for the general advantage estimation.
    Must be 1.0 when gamma=1.0: GAE(lambda<1) produces biased returns for
    terminal-only reward, making explained variance artificially low even
    with a perfect value function."""
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    log_interval: int = 81_920
    """the interval (in samples) at which to run benchmark (~20 iterations)"""
    save_interval: int = 81_920
    """the interval (in samples) at which to save the model (~20 iterations)"""
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks : Optional[torch.BoolTensor] = None):
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

# ── EcoAgent: R-öko actor-critic with LSTM ──────────────────────────────────

class EcoAgent(nn.Module):
    """
    R-öko actor-critic with LSTM. Pure/stateless: hidden state passed in/out.

    Architecture: shared_encode → pre_lstm → LSTM → separate actor/critic heads.
    """
    EMB_DIM = 32
    HIDDEN  = 256

    LSTM_HIDDEN = 128
    LSTM_LAYERS = 1

    def __init__(self, num_players: int = 2, score_shortcut: bool = False,
                 lstm_hidden: int = 128, lstm_layers: int = 1):
        super().__init__()
        E, H = self.EMB_DIM, self.HIDDEN
        self.num_players = num_players
        self.score_shortcut = score_shortcut
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        float_dim = eco_float_dim(num_players)

        self.player_emb = nn.Embedding(num_players + 1, E)   # tokens 1..num_players
        self.phase_emb  = nn.Embedding(2, E)                 # 0=play, 1=discard

        # Shared encoder: float features → H
        self.flat_enc = nn.Sequential(
            layer_init(nn.Linear(float_dim, H)), nn.LayerNorm(H), nn.ReLU(),
            layer_init(nn.Linear(H, H)),         nn.LayerNorm(H), nn.ReLU(),
        )
        fusion_in = H + 2 * E

        # Pre-LSTM projection
        self.pre_lstm = nn.Sequential(
            layer_init(nn.Linear(fusion_in, H)), nn.LayerNorm(H), nn.ReLU(),
        )

        # LSTM (shared between actor and critic)
        self.lstm = nn.LSTM(H, lstm_hidden, num_layers=lstm_layers)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # Separate actor and critic trunks after LSTM
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden, H)), nn.LayerNorm(H), nn.ReLU(),
        )
        self.critic_trunk = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden, H)), nn.LayerNorm(H), nn.ReLU(),
        )
        self.actor_head  = layer_init(nn.Linear(H, NUM_ACTIONS), std=0.01)

        if score_shortcut:
            # Shallow path: scores → small hidden → 1 scalar
            self.score_branch = nn.Sequential(
                layer_init(nn.Linear(num_players, 32)), nn.ReLU(),
                layer_init(nn.Linear(32, 1), std=1.0),
            )
            # Deep path output + shallow path output → final value
            self.critic_head = layer_init(nn.Linear(H + 1, 1), std=1.0)
        else:
            self.critic_head = layer_init(nn.Linear(H, 1), std=1.0)

    def _shared_encode(self, obs: EcoPyTreeObs) -> torch.Tensor:
        # Embed discrete tokens
        player_repr = self.player_emb(obs.current_player.squeeze(-1))   # (B, E)
        phase_repr  = self.phase_emb(obs.phase.squeeze(-1))             # (B, E)

        # Encode all float features
        flat = torch.cat([
            obs.hands, obs.recycling_side, obs.waste_side,
            obs.factory_stacks, obs.collected,
            obs.penalty_pile, obs.scores, obs.draw_pile_size,
            obs.draw_pile_comp,
        ], dim=-1)
        flat_repr = self.flat_enc(flat)                                  # (B, H)

        return torch.cat([flat_repr, player_repr, phase_repr], dim=-1)   # (B, fusion_in)

    def get_states(self, obs: EcoPyTreeObs, lstm_state: LSTMState, done: torch.Tensor):
        """
        Encode obs through shared encoder + LSTM, resetting hidden on done.
        obs shape: (T*B, ...) or (B, ...); lstm_state batch dim = B; done: (T*B,) or (B,).
        Returns (hidden (T*B, lstm_hidden), new_lstm_state).
        """
        shared = self._shared_encode(obs)
        hidden = self.pre_lstm(shared)

        # LSTM: process sequentially, resetting state on episode boundaries
        batch_size = lstm_state.h.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        h, c = lstm_state.h, lstm_state.c
        for t_h, t_d in zip(hidden, done):
            h = (1.0 - t_d).view(1, -1, 1) * h
            c = (1.0 - t_d).view(1, -1, 1) * c
            t_h, (h, c) = self.lstm(t_h.unsqueeze(0), (h, c))
            new_hidden.append(t_h)
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, LSTMState(h=h, c=c)

    def _critic_value(self, obs: EcoPyTreeObs, critic_h: torch.Tensor) -> torch.Tensor:
        if self.score_shortcut:
            score_val = self.score_branch(obs.scores)   # (B, 1)
            return self.critic_head(torch.cat([critic_h, score_val], dim=-1))
        return self.critic_head(critic_h)

    def get_value(self, obs: EcoPyTreeObs, lstm_state: LSTMState, done: torch.Tensor):
        hidden, _ = self.get_states(obs, lstm_state, done)
        critic_h = self.critic_trunk(hidden)
        return self._critic_value(obs, critic_h)

    def get_action_and_value(self, obs: EcoPyTreeObs, action_mask,
                             lstm_state: LSTMState, done: torch.Tensor,
                             action=None):
        hidden, new_lstm_state = self.get_states(obs, lstm_state, done)
        actor_h = self.actor_trunk(hidden)
        logits = self.actor_head(actor_h)
        probs  = CategoricalMasked(logits=logits, masks=action_mask)
        if action is None:
            action = probs.sample()
            # Safety: force legal if sampling produced illegal action
            illegal = ~action_mask.gather(1, action.unsqueeze(1)).squeeze(1)
            if illegal.any():
                n = illegal.sum().item()
                print(f"[WARN] Hard mask enforcement triggered for {n}/{len(action)} actions")
                masked_logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=logits.device))
                fallback = masked_logits.argmax(dim=1)
                action = torch.where(illegal, fallback, action)
        critic_h = self.critic_trunk(hidden)
        return action, probs.log_prob(action), probs.entropy(), self._critic_value(obs, critic_h), new_lstm_state

# ── Player interface: abstract base for batched opponent players ─────────────

class BasePlayer(ABC):
    """
    Abstract interface for opponent players used by VecSinglePlayerEcoEnv.

    - batch_action(obs, mask, idxs): batched inference for a subset of envs
    - reset(env_indices): reset internal state for given envs (or all)
    - slice(i): returns an object with action(obs, mask) -> int for per-env use
    """
    @abstractmethod
    def batch_action(self, obs_batch, mask_batch, idxs: list) -> np.ndarray:
        ...

    @abstractmethod
    def reset(self, env_indices=None) -> None:
        ...

    @abstractmethod
    def slice(self, env_idx: int) -> Any:
        """Return an object with action(obs, mask) -> int."""
        ...


class RandomBatchedPlayer(BasePlayer):
    """Random opponent. Stateless."""

    def batch_action(self, obs_batch, mask_batch, idxs=None) -> np.ndarray:
        return np.array([
            int(np.random.choice(np.where(m)[0])) for m in mask_batch
        ], dtype=np.int32)

    def reset(self, env_indices=None) -> None:
        pass

    def slice(self, env_idx: int) -> "RandomBatchedPlayer":
        return self

    def action(self, obs, mask) -> int:
        return int(np.random.choice(np.where(mask)[0]))


# ── BatchedPlayer / SlicedPlayer: LSTM opponent ─────────────────────────────

class BatchedPlayer(BasePlayer):
    """
    Manages batched LSTM hidden states for N envs.
    The NN agent is stateless (pure); this class holds the recurrent state.
    """
    def __init__(self, agent: EcoAgent, device, num_envs: int):
        self.agent = agent
        self.device = device
        self.num_envs = num_envs
        self.lstm_state = make_lstm_state(
            agent.lstm_layers, num_envs, agent.lstm_hidden, device
        )

    def reset(self, env_indices=None):
        """Reset LSTM state for given env indices (or all if None)."""
        if env_indices is None:
            self.lstm_state = make_lstm_state(
                self.agent.lstm_layers, self.num_envs, self.agent.lstm_hidden, self.device
            )
        else:
            for idx in env_indices:
                self.lstm_state.h[:, idx] = 0
                self.lstm_state.c[:, idx] = 0

    def batch_action(self, obs_batch: EcoPyTreeObs, mask_batch: np.ndarray,
                     idxs: list) -> np.ndarray:
        """Batched action for a subset of envs identified by idxs."""
        obs_t = obs_to_tensor(obs_batch, self.device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=self.device)
        done_t = torch.zeros(len(idxs), device=self.device)
        # Slice LSTM state to only the active env indices
        idx_t = torch.tensor(idxs, dtype=torch.long, device=self.device)
        sub_state = LSTMState(
            h=self.lstm_state.h[:, idx_t].contiguous(),
            c=self.lstm_state.c[:, idx_t].contiguous(),
        )
        with torch.no_grad():
            acts, _, _, _, new_state = self.agent.get_action_and_value(
                obs_t, mask_t, sub_state, done_t
            )
        # Write updated LSTM state back to the correct indices
        self.lstm_state.h[:, idx_t] = new_state.h
        self.lstm_state.c[:, idx_t] = new_state.c
        return acts.cpu().numpy()

    def slice(self, env_idx: int) -> "SlicedPlayer":
        return SlicedPlayer(self, env_idx)


class SlicedPlayer:
    """
    Per-env proxy into a BatchedPlayer. Passed to SinglePlayerEcoEnv as opponent_fn.
    action() is sync: slices the batched LSTM state, runs one forward pass, writes back.
    Used during reset() and any per-env fallback path.
    """
    def __init__(self, batched: BatchedPlayer, env_idx: int):
        self.batched = batched
        self.env_idx = env_idx

    def action(self, obs, mask) -> int:
        i = self.env_idx
        player = self.batched
        obs_t = obs_to_tensor(EcoPyTreeObs(*[np.expand_dims(f, 0) for f in obs]), player.device)
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


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def _select_checkpoint_paths(model_dir: str, exclude_latest: bool = True) -> list:
    """Select checkpoints with exponential spacing: most recent 1, 2, 4, 8, ... back.

    Returns list of (label, path) sorted from newest to oldest."""
    files = glob.glob(os.path.join(model_dir, "eco_*.pkt"))
    # Filter out eco_latest.pkt
    files = [f for f in files if "latest" not in os.path.basename(f)]
    if not files:
        return []
    # Sort by step number (filename eco_{step}.pkt)
    def _step(f):
        base = os.path.basename(f).replace("eco_", "").replace(".pkt", "")
        try: return int(base)
        except ValueError: return 0
    files.sort(key=_step, reverse=True)  # newest first
    if exclude_latest and len(files) > 0:
        files = files[1:]  # skip the most recent (it's the current agent)
    selected = []
    idx = 0
    gap = 1
    while idx < len(files):
        step = _step(files[idx])
        selected.append((f"vs_ckpt_{step}", files[idx]))
        idx += gap
        gap *= 2
    return selected


def benchmark(_agent, num_players: int = 2, num_games: int = 200,
              device="cuda", model_dir: str = "model", tournament: bool = True) -> dict:
    """
    Benchmark the trained agent against random opponents and past checkpoints.
    All agents (including random) use the unified BatchedPlayer interface.

    Returns
    -------
    dict[scenario_name -> {"rewards", "all_scores", "agent_scores"}]
    """
    num_envs = min(num_games, 128)

    # Agent under test — BatchedPlayer for batched inference + state tracking
    agent_player = BatchedPlayer(_agent, device, num_envs=num_envs)

    def _run(opp_player):
        """Run num_games with agent_player vs opp_player, collect agent seat rewards."""
        agent_player.reset()
        vec = VecSinglePlayerEcoEnv(num_envs=num_envs, num_players=num_players,
                                     opponent=opp_player)
        obs_np, masks_np = vec.reset()
        rewards_list = []
        scores_list  = []
        seats_list   = []
        all_idxs = list(range(num_envs))
        while len(rewards_list) < num_games:
            actions = agent_player.batch_action(obs_np, masks_np, all_idxs)
            obs_np, masks_np, _, terminated, truncated, infos = vec.step(actions)
            done_mask = np.logical_or(terminated, truncated)
            done_idxs = list(np.where(done_mask)[0])
            if done_idxs:
                agent_player.reset(done_idxs)
            for i in done_idxs:
                if len(rewards_list) >= num_games:
                    break
                scores = infos[i]["final_scores"]
                seat   = infos[i]["agent_seat"]
                scores_list.append(scores.copy())
                seats_list.append(seat)
                rewards_list.append(float(scores[seat]) / MAX_ECO_SCORE)
        vec.close()
        rewards      = np.array(rewards_list, dtype=np.float32)
        all_scores   = np.stack(scores_list)                          # (N, num_players)
        agent_scores = np.array([s[seats_list[j]] for j, s in enumerate(scores_list)], dtype=np.float32)
        return rewards, all_scores, agent_scores

    scenarios: dict[str, BasePlayer] = {
        "vs_random": RandomBatchedPlayer(),
    }

    # Add tournament vs past checkpoints (exponential spacing)
    if tournament:
        ckpts = _select_checkpoint_paths(model_dir)
        for label, path in ckpts:
            loaded = False
            for sc in [False, True]:
                try:
                    old_agent = EcoAgent(num_players=num_players, score_shortcut=sc).to(device)
                    old_agent.load_state_dict(torch.load(path, map_location=device))
                    old_agent.eval()
                    scenarios[label] = BatchedPlayer(old_agent, device, num_envs=num_envs)
                    loaded = True
                    break
                except Exception:
                    continue
            if not loaded:
                print(f"[benchmark] Skipping {path}: incompatible checkpoint")

    # Pre-collect so we can pair (rewards, all_scores) correctly
    out = {}
    for name, player in scenarios.items():
        rewards, all_scores, agent_scores = _run(player)
        out[name] = {"rewards": rewards, "all_scores": all_scores, "agent_scores": agent_scores}
    return out


_SCENARIO_NOTES = {
    "vs_random":       "agent vs random opponent",
}


def log_benchmark(results: dict, global_step: int) -> None:
    """
    Log benchmark results to wandb and print a summary table.

    results : dict[scenario -> {"rewards": (N,) float32, "scores": (N,) float32}]
    Win = highest score at the table (ties count as win for all tied players).
    """
    hdr = f"  {'scenario':<18}  {'win%':>6}  {'avg score':>10}  {'score std':>10}  {'avg reward':>11}  {'reward std':>11}  note"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    log_dict: dict[str, Any] = {"global_step": global_step}

    for name, data in results.items():
        rewards    = data["rewards"]
        all_scores = data["all_scores"]          # (N, P) — one row per game
        max_scores = all_scores.max(axis=1)      # (N,) best score at the table
        agent_scores = data["agent_scores"]
        wins = (agent_scores >= max_scores).astype(np.float32)
        win_rate     = float(wins.mean()) * 100
        mean_score   = float(agent_scores.mean())
        std_score    = float(agent_scores.std())
        mean_reward  = float(rewards.mean())
        std_reward   = float(rewards.std())
        note = _SCENARIO_NOTES.get(name, "vs past checkpoint" if name.startswith("vs_ckpt_") else "")
        print(f"  {name:<18}  {win_rate:>5.1f}%  {mean_score:>10.2f}  {std_score:>10.2f}  {mean_reward:>+11.4f}  {std_reward:>11.4f}  {note}")
        log_dict[f"benchmark/{name}/win_rate"]    = win_rate
        log_dict[f"benchmark/{name}/mean_score"]  = mean_score
        log_dict[f"benchmark/{name}/std_score"]   = std_score
        log_dict[f"benchmark/{name}/mean_reward"] = mean_reward
        log_dict[f"benchmark/{name}/std_reward"]  = std_reward

    if wandb.run is not None:
        wandb.log(log_dict)

    print()

if __name__ == "__main__":
    args = tyro.cli(Args)
    import wandb
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.use_deterministic_algorithms(args.torch_deterministic)
    if args.torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Agent (pure/stateless — LSTM hidden state managed externally)
    agent = EcoAgent(num_players=args.num_players, score_shortcut=args.score_shortcut).to(device)

    # Opponent setup: BatchedPlayer handles both batched and per-env (sliced) inference.
    # VecSinglePlayerEcoEnv wires it up: batch_action for step(), slice(i).action for reset().
    if args.opponent_mode == "self_play":
        opponent = BatchedPlayer(agent, device, num_envs=args.num_envs)
    else:  # random
        opponent = RandomBatchedPlayer()

    # env setup
    envs = VecSinglePlayerEcoEnv(num_envs=args.num_envs, num_players=args.num_players,
                                  opponent=opponent,
                                  reward_shaping_scale=args.reward_shaping_scale,
                                  opponent_penalty=args.opponent_penalty,
                                  relative_seat=args.relative_seat)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # [PYTREE] obs is a PyTreeObs buffer; each leaf is (num_steps, num_envs, leaf_dim)
    _proto_obs, _proto_masks = envs.reset(seed=args.seed)
    obs          = alloc_obs_buffer(_proto_obs, args.num_steps, args.num_envs, device)
    actions      = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    action_masks   = torch.zeros((args.num_steps, args.num_envs, NUM_ACTIONS), dtype=torch.bool).to(device)

    # [PYTREE] obs already fetched above for prototype; convert to tensor
    next_obs  = obs_to_tensor(_proto_obs, device)
    next_masks = torch.as_tensor(_proto_masks, dtype=torch.bool, device=device)
    next_done = torch.zeros(args.num_envs).to(device)
    # LSTM state for the training agent (carried across steps, reset on done inside get_states)
    next_lstm_state = make_lstm_state(agent.lstm_layers, args.num_envs, agent.lstm_hidden, device)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    last_log_step = 0
    last_save_step = 0
    start_time = time.time()
    alpha = args.learning_rate

    # Baseline benchmark skipped — too slow for untrained agents

    for iteration in range(1, args.num_iterations + 1):
        # Save initial LSTM state for training (needed to recompute hidden states)
        initial_lstm_state = LSTMState(
            h=next_lstm_state.h.clone(),
            c=next_lstm_state.c.clone(),
        )
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            assert args.target_kl is None, "Cannot anneal learning rate when target_kl is set."
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if args.target_kl is not None:
            optimizer.param_groups[0]["lr"] = alpha

        # Entropy coefficient annealing: linear decay from ent_coef → ent_coef_end
        # over the first ent_anneal_steps, then hold at ent_coef_end
        if args.ent_coef_end > 0 and args.ent_anneal_steps > 0:
            steps_so_far = (iteration - 1) * args.batch_size
            frac = min(steps_so_far / args.ent_anneal_steps, 1.0)
            ent_coef_now = args.ent_coef + frac * (args.ent_coef_end - args.ent_coef)
        else:
            ent_coef_now = args.ent_coef

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            # [PYTREE] store obs: tree_map assigns each leaf buffer's step slot
            tree_map(lambda buf, val: buf.__setitem__(step, val), obs, next_obs)
            dones[step] = next_done
            action_masks[step] = next_masks

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs, action_masks[step], next_lstm_state, next_done
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_np, next_masks_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            # Reset opponent LSTM state for terminated envs
            done_indices = list(np.where(next_done)[0])
            if done_indices:
                opponent.reset(done_indices)
            # reward is a scalar per env
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            # [PYTREE] convert numpy pytree obs → tensor pytree obs
            next_obs   = obs_to_tensor(next_obs_np, device)
            next_masks = torch.as_tensor(next_masks_np, dtype=torch.bool, device=device)
            next_done  = torch.Tensor(next_done).to(device)

            for i in range(len(infos)):
                if infos[i].get("final_scores") is not None:
                    ep_return = float(reward[i])
                    if wandb.run is not None:
                        wandb.log({"charts/episodic_return": ep_return, "global_step": global_step})

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        # [PYTREE] flatten time+env dims for each leaf independently
        b_obs = tree_map(lambda x: x.reshape(-1, *x.shape[2:]), obs)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_action_masks = action_masks.reshape((-1, action_masks.shape[-1]))

        # Optimizing the policy and value network
        # LSTM PPO: minibatch by env index (not random), preserving temporal order
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # preserves temporal order

                # [PYTREE] index each leaf by minibatch indices
                mb_obs = tree_map(lambda x: x[mb_inds], b_obs)
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    mb_obs, b_action_masks[mb_inds],
                    LSTMState(
                        h=initial_lstm_state.h[:, mbenvinds],
                        c=initial_lstm_state.c[:, mbenvinds],
                    ),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        # Apply adaptive learning rate adjustment based on KL divergence
        if args.target_kl is not None:
            # currently we just use approx_kl as a proxy to the true KL divergence
            # but it should be sufficient for our simple environments. For more complex environments, consider using the true KL divergence.
            global_approx_kl = approx_kl
            if global_approx_kl > 2.0 * args.target_kl:
                alpha = max(1e-5, alpha / 1.5)
            elif global_approx_kl < 0.5 * args.target_kl:
                alpha = min(1e-2, alpha * 1.5)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if wandb.run is not None:
            wandb.log({
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "global_step": global_step,
            })


        # Inside the training loop:
        if global_step >= last_log_step + args.log_interval and global_step > 0:
            bench_results = benchmark(agent, args.num_players, device=str(device), model_dir=args.model_dir, tournament=False)
            log_benchmark(bench_results, global_step)
            last_log_step = global_step

        if global_step >= last_save_step + args.save_interval and global_step > 0:
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(agent.state_dict(), os.path.join(args.model_dir, f"eco_{global_step}.pkt"))
            torch.save(agent.state_dict(), os.path.join(args.model_dir, "eco_latest.pkt"))
            last_save_step = global_step
        print(f"Step: {global_step} SPS:", int(global_step / (time.time() - start_time)))
        if wandb.run is not None:
            wandb.log({"charts/SPS": int(global_step / (time.time() - start_time)), "global_step": global_step})

    envs.close()
    if wandb.run is not None:
        wandb.finish()
