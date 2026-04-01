# eco_ppo.py — PPO training for R-öko.
# Derived from ppo.py with minimal changes:
#   - Imports eco_vec_env / eco_obs_encoder instead of vec_env / obs_encoder
#   - EcoAgent replaces Agent (flat-obs architecture, 108 actions)
#   - Hearts-specific heuristic agents removed (no equivalent in R-öko)
#   - Benchmark simplified to vs_random only
#   - Batch opponent functions added for async vec env stepping
import glob
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from eco_vec_env import VecSinglePlayerEcoEnv
from eco_obs_encoder import EcoPyTreeObs, SinglePlayerEcoEnv, eco_float_dim
from eco_env import NUM_ACTIONS, NUM_COLORS, MAX_ECO_SCORE

# ── PyTree helpers (only 6 lines in the training loop touch obs) ─────────────
import torch
from torch.utils._pytree import tree_map

def _obs_dtype(x: "np.ndarray"):
    """Return the torch dtype matching a PyTreeObs leaf by inspecting its numpy dtype."""
    import numpy as np
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
    wandb: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ppo-eco"
    """the wandb's project name"""
    wandb_entity: str = None
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
    opponent_mode: Literal["self_play", "random", "mixed"] = "self_play"
    """opponent policy used during training:
      self_play  — opponent uses the current agent weights (default)
      random     — opponent plays uniformly at random
      mixed      — half the envs use self-play, half use random opponents"""
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
    log_interval: int = 10_000
    """the interval (in steps) at which to log training progress"""
    save_interval: int = 20_000
    """the interval (in steps) at which to save the model"""
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

from typing import *
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks : Optional[torch.BoolTensor] = None):
        self.masks = masks
        if masks is None:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            _masks = masks.to(probs.device if probs is not None else logits.device)
            self.masks = _masks
            assert logits is not None
            logits = torch.where(_masks, logits, -1e8)
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if self.masks is None:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, 0)
        return -p_log_p.sum(-1)

# ── EcoAgent (replaces Hearts Agent — different obs structure) ───────────────

class EcoAgent(nn.Module):
    """
    R-öko actor-critic. Accepts an EcoPyTreeObs of tensors.

    Embedding index fields (long):
      current_player, phase

    Continuous fields (float32):
      hands, recycling_side, waste_side, factory_stacks, collected,
      penalty_pile, draw_pile_size
    """
    EMB_DIM = 32
    HIDDEN  = 128

    def __init__(self, num_players: int = 2):
        super().__init__()
        E, H = self.EMB_DIM, self.HIDDEN
        self.num_players = num_players
        float_dim = eco_float_dim(num_players)

        self.player_emb = nn.Embedding(num_players + 1, E)   # tokens 1..num_players
        self.phase_emb  = nn.Embedding(2, E)                 # 0=play, 1=discard

        self.flat_enc = nn.Sequential(
            layer_init(nn.Linear(float_dim, H)), nn.LayerNorm(H), nn.ReLU(),
        )
        fusion_in = H + 2 * E
        self.fusion = nn.Sequential(
            layer_init(nn.Linear(fusion_in, H * 2)), nn.LayerNorm(H * 2), nn.ReLU(),
            layer_init(nn.Linear(H * 2, H)),         nn.LayerNorm(H),     nn.ReLU(),
        )
        self.actor_head  = layer_init(nn.Linear(H, NUM_ACTIONS), std=0.01)
        self.critic_head = layer_init(nn.Linear(H, 1),           std=1.0)

    def _encode(self, obs: EcoPyTreeObs) -> torch.Tensor:
        # Embed discrete tokens
        player_repr = self.player_emb(obs.current_player.squeeze(-1))   # (B, E)
        phase_repr  = self.phase_emb(obs.phase.squeeze(-1))             # (B, E)

        # Encode all float features
        flat = torch.cat([
            obs.hands, obs.recycling_side, obs.waste_side,
            obs.factory_stacks, obs.collected,
            obs.penalty_pile, obs.scores, obs.draw_pile_size,
        ], dim=-1)
        flat_repr = self.flat_enc(flat)                                  # (B, H)

        return self.fusion(torch.cat([flat_repr, player_repr, phase_repr], dim=-1))

    def get_value(self, obs: EcoPyTreeObs):
        return self.critic_head(self._encode(obs))

    def get_action_and_value(self, obs: EcoPyTreeObs, action_mask, action=None):
        hidden = self._encode(obs)
        logits = self.actor_head(hidden)
        probs  = CategoricalMasked(logits=logits, masks=action_mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_head(hidden)

# ── Random agent ─────────────────────────────────────────────────────────────

class RandomAgent:
    """
    Selects a uniformly random legal action each step.
    Stateless — acts only from the action mask, needs no observation.
    """
    def select_actions(self, masks: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        masks : (N, 108) bool  legal-action masks

        Returns
        -------
        actions : (N,) int
        """
        actions = np.empty(len(masks), dtype=np.int32)
        for i, m in enumerate(masks):
            actions[i] = np.random.choice(np.where(m)[0])
        return actions


# (No heuristic agents for R-öko — only RandomAgent as baseline)


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def _as_opponent_fn(agent, device) -> callable:
    """
    Unified adapter: converts any agent type into a SinglePlayerEnv opponent_fn.

    Supports:
      - EcoAgent (nn.Module with get_action_and_value)
      - RandomAgent (anything with select_actions(masks) -> int array)
    """
    if hasattr(agent, 'select_actions'):
        def fn(obs, mask):
            return int(agent.select_actions(mask[np.newaxis])[0])
        return fn
    # nn.Module agent
    def fn(obs, mask):
        obs_t  = obs_to_tensor(EcoPyTreeObs(*[np.expand_dims(f, 0) for f in obs]), device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
        return int(action.item())
    return fn


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
              device="cuda", model_dir: str = "model") -> dict:
    """
    Benchmark the trained agent against random opponents and past checkpoints.

    Returns
    -------
    dict[scenario_name -> {"rewards", "all_scores", "agent_scores"}]
    """
    num_envs = min(num_games, 128)

    # Batched action selector for the agent under test
    if hasattr(_agent, 'get_action_and_value'):
        def _select(obs_np, masks_np):
            obs_t  = obs_to_tensor(obs_np, device)
            mask_t = torch.as_tensor(masks_np, dtype=torch.bool, device=device)
            with torch.no_grad():
                actions, _, _, _ = _agent.get_action_and_value(obs_t, mask_t)
            return actions.cpu().numpy()
    else:
        def _select(obs_np, masks_np):
            return _agent.select_actions(masks_np)

    def _run(opponent_fn):
        """Run num_games with _agent vs opponent_fn, collect agent seat rewards."""
        vec = VecSinglePlayerEcoEnv(num_envs=num_envs, num_players=num_players,
                                     opponent_fn=opponent_fn)
        obs_np, masks_np = vec.reset()
        rewards_list = []
        scores_list  = []
        seats_list   = []
        while len(rewards_list) < num_games:
            actions = _select(obs_np, masks_np)
            obs_np, masks_np, _, terminated, truncated, infos = vec.step(actions)
            for i in np.where(np.logical_or(terminated, truncated))[0]:
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

    scenarios = {
        "vs_random": _as_opponent_fn(RandomAgent(), device),
    }

    # Add tournament vs past checkpoints (exponential spacing)
    ckpts = _select_checkpoint_paths(model_dir)
    for label, path in ckpts:
        try:
            old_agent = EcoAgent(num_players=num_players).to(device)
            old_agent.load_state_dict(torch.load(path, map_location=device))
            old_agent.eval()
            scenarios[label] = _as_opponent_fn(old_agent, device)
        except Exception as e:
            print(f"[benchmark] Skipping {path}: {e}")

    # Pre-collect so we can pair (rewards, all_scores) correctly
    out = {}
    for name, fn in scenarios.items():
        rewards, all_scores, agent_scores = _run(fn)
        out[name] = {"rewards": rewards, "all_scores": all_scores, "agent_scores": agent_scores}
    return out


_SCENARIO_NOTES = {
    "vs_random":       "agent vs random opponent",
}


def log_benchmark(results: dict, writer, global_step: int) -> None:
    """
    Write benchmark results to tensorboard and print a summary table.
    Also logs a wandb.Table if wandb is active (dynamic rows for checkpoint scenarios).

    results : dict[scenario -> {"rewards": (N,) float32, "scores": (N,) float32}]
    Win = highest score at the table (ties count as win for all tied players).
    """
    hdr = f"  {'scenario':<18}  {'win%':>6}  {'avg score':>10}  {'score std':>10}  {'avg reward':>11}  {'reward std':>11}  note"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    # Collect rows for wandb table
    table_rows = []

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
        writer.add_scalar(f"benchmark/{name}/win_rate",    win_rate,    global_step)
        writer.add_scalar(f"benchmark/{name}/mean_score",  mean_score,  global_step)
        writer.add_scalar(f"benchmark/{name}/std_score",   std_score,   global_step)
        writer.add_scalar(f"benchmark/{name}/mean_reward", mean_reward, global_step)
        writer.add_scalar(f"benchmark/{name}/std_reward",  std_reward,  global_step)
        table_rows.append([name, win_rate, mean_score, std_score, mean_reward, note])

    # Log wandb table for tournament results (handles dynamic checkpoint rows)
    try:
        import wandb
        if wandb.run is not None:
            table = wandb.Table(
                columns=["scenario", "win%", "avg_score", "score_std", "avg_reward", "note"],
                data=table_rows,
            )
            wandb.log({"benchmark/tournament": table, "global_step": global_step})
    except ImportError:
        pass

    print()

if __name__ == "__main__":
    args = tyro.cli(Args)
    # patch tensorboard if not installed
    # class DummyWriter:
    #     def __init__(self, *args, **kwargs): pass
    #     def add_scalar(self, *args, **kwargs): pass
    #     def add_text(self, *args, **kwargs): pass
    #     def close(self): pass
    # try:
    #     if args.track:
    #         from torch.utils.tensorboard import SummaryWriter
    #     else:
    #         SummaryWriter = DummyWriter
    # except ImportError:
    #     SummaryWriter = DummyWriter
    from torch.utils.tensorboard import SummaryWriter
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track and args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Agent first — self-play opponent_fn closes over it and tracks weights live
    agent = EcoAgent(num_players=args.num_players).to(device)

    def _self_play_fn(obs, mask):
        """Opponent uses the current training agent weights."""
        obs_t  = obs_to_tensor(
            EcoPyTreeObs(*[np.expand_dims(f, 0) for f in obs]), device
        )
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
        return int(action.item())

    def _random_fn(obs, mask):
        """Opponent plays uniformly at random over legal cards."""
        return int(np.random.choice(np.where(mask)[0]))

    # Batched opponent fns: evaluate ALL pending opponent obs in one call.
    # Passed to envs.step() so the vec env can batch across games.
    def _batch_self_play_fn(obs_batch: EcoPyTreeObs, mask_batch: np.ndarray) -> np.ndarray:
        obs_t  = obs_to_tensor(obs_batch, device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=device)
        with torch.no_grad():
            acts, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
        return acts.cpu().numpy()

    def _batch_random_fn(obs_batch: EcoPyTreeObs, mask_batch: np.ndarray) -> np.ndarray:
        return np.array([
            int(np.random.choice(np.where(m)[0])) for m in mask_batch
        ], dtype=np.int32)

    if args.opponent_mode == "self_play":
        opponent_fn = _self_play_fn
        batch_opp_fn = _batch_self_play_fn
        n_self  = args.num_envs
        n_rand  = 0
    elif args.opponent_mode == "random":
        opponent_fn = _random_fn
        batch_opp_fn = _batch_random_fn
        n_self  = 0
        n_rand  = args.num_envs
    else:  # mixed
        n_self  = args.num_envs // 2
        n_rand  = args.num_envs - n_self
        opponent_fn = None  # assigned per-env below
        batch_opp_fn = None  # handled inside _MixedVecEnv

    # env setup
    if args.opponent_mode == "mixed":
        self_envs = VecSinglePlayerEcoEnv(num_envs=n_self, num_players=args.num_players,
                                       opponent_fn=_self_play_fn,
                                       reward_shaping_scale=args.reward_shaping_scale)
        rand_envs = VecSinglePlayerEcoEnv(num_envs=n_rand, num_players=args.num_players,
                                       opponent_fn=_random_fn,
                                       reward_shaping_scale=args.reward_shaping_scale)

        class _MixedVecEnv:
            """Thin combiner: concatenates self-play and random envs along the batch axis."""
            def __init__(self, a, b):
                self._a, self._b = a, b
                self.num_envs = a.num_envs + b.num_envs
            def reset(self, seed=None):
                oa, ma = self._a.reset(seed=seed)
                ob, mb = self._b.reset(seed=(seed + n_self) if seed is not None else None)
                return _stack(oa, ob), np.concatenate([ma, mb], axis=0)
            def step(self, actions, batch_opponent_fn=None):
                aa, ab = actions[:n_self], actions[n_self:]
                oa, ma, ra, ta, ua, ia = self._a.step(aa, batch_opponent_fn=_batch_self_play_fn)
                ob, mb, rb, tb, ub, ib = self._b.step(ab, batch_opponent_fn=_batch_random_fn)
                return _stack(oa, ob), np.concatenate([ma, mb], axis=0), \
                       np.concatenate([ra, rb]), np.concatenate([ta, tb]), \
                       np.concatenate([ua, ub]), ia + ib
            def close(self):
                self._a.close(); self._b.close()

        def _stack(a, b):
            return EcoPyTreeObs(*[np.concatenate([getattr(a, f), getattr(b, f)], axis=0)
                               for f in EcoPyTreeObs._fields])

        envs = _MixedVecEnv(self_envs, rand_envs)
    else:
        envs = VecSinglePlayerEcoEnv(num_envs=args.num_envs, num_players=args.num_players,
                                  opponent_fn=opponent_fn,
                                  reward_shaping_scale=args.reward_shaping_scale)
    # [PYTREE] No single_observation_space needed; agent is shape-agnostic

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
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    last_log_step = 0
    last_save_step = 0
    start_time = time.time()
    alpha = args.learning_rate

    # Baseline benchmark: random agent and untrained agent before any optimization
    print("=== Baseline benchmark (before training) ===")
    random_results = benchmark(RandomAgent(), args.num_players, device=device, model_dir="model")
    log_benchmark(random_results, writer, global_step=0)
    untrained_results = benchmark(agent, args.num_players, device=device, model_dir="model")
    log_benchmark(untrained_results, writer, global_step=0)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            assert args.target_kl is None, "Cannot anneal learning rate when target_kl is set."
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if args.target_kl is not None:
            optimizer.param_groups[0]["lr"] = alpha

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            # [PYTREE] store obs: tree_map assigns each leaf buffer's step slot
            tree_map(lambda buf, val: buf.__setitem__(step, val), obs, next_obs)
            dones[step] = next_done
            action_masks[step] = next_masks

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, action_masks[step])
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_np, next_masks_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy(), batch_opponent_fn=batch_opp_fn)
            next_done = np.logical_or(terminations, truncations)
            # reward is a scalar per env
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            # [PYTREE] convert numpy pytree obs → tensor pytree obs
            next_obs   = obs_to_tensor(next_obs_np, device)
            next_masks = torch.as_tensor(next_masks_np, dtype=torch.bool, device=device)
            next_done  = torch.Tensor(next_done).to(device)

            for i in range(len(infos)):
                if infos[i].get("final_scores") is not None:
                    ep_return = float(reward[i])
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_action_masks = action_masks.reshape((-1, action_masks.shape[-1]))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # [PYTREE] index each leaf by minibatch indices
                mb_obs = tree_map(lambda x: x[mb_inds], b_obs)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_action_masks[mb_inds], b_actions.long()[mb_inds])
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

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
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)


        # Inside the training loop:
        if global_step >= last_log_step + args.log_interval and global_step > 0:
            bench_results = benchmark(agent, args.num_players, device=device, model_dir="model")
            log_benchmark(bench_results, writer, global_step)
            last_log_step = global_step

        if global_step >= last_save_step + args.save_interval and global_step > 0:
            os.makedirs("model", exist_ok=True)
            torch.save(agent.state_dict(), f"model/eco_{global_step}.pkt")
            torch.save(agent.state_dict(), "model/eco_latest.pkt")
            last_save_step = global_step
        print(f"Step: {global_step} SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
