"""
Vectorized single-player environment — runs N SinglePlayerEnvs in lockstep.

When a BatchStepper is provided, env stepping runs entirely in compiled Numba
code (one Python→JIT transition per batch call). Only escapes to Python for
neural network inference (opponent actions).

Falls back to a generic Python loop for engines without BatchStepper.

Interface:
    obs, masks = vec.reset()
    obs, masks, rewards, terminated, truncated, infos = vec.step(actions)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from .game import BaseGameEngine
from .single_player_env import SinglePlayerEnv
from .player import BasePlayer
from .key import Key
from .batch_stepper import BatchStepper


def _alloc_obs_buf(template_obs, N: int):
    """Allocate a batched obs buffer (N copies) from a single-obs template."""
    cls = type(template_obs)
    return cls(**{
        field: np.zeros((N,) + getattr(template_obs, field).shape,
                        dtype=getattr(template_obs, field).dtype)
        for field in cls._fields
    })


def _slice_obs(buf, n: int):
    """Return a view of the first n entries of a batched obs buffer."""
    cls = type(buf)
    return cls(**{field: getattr(buf, field)[:n] for field in cls._fields})


class EnvFactory(ABC):
    @abstractmethod
    def create(self, rng: np.random.Generator) -> BaseGameEngine:
        ...


class VecSinglePlayerEnv:
    """
    Synchronous vectorised single-player environment.

    Uses BatchStepper for JIT-accelerated stepping when provided.
    Falls back to generic per-env stepping otherwise.
    """

    def __init__(
        self,
        num_envs: int,
        opponent: BasePlayer,
        env_factory: EnvFactory,
        key: Key,
        batch_stepper: Optional[BatchStepper] = None,
    ):
        self.num_envs = num_envs
        self._opponent = opponent
        self._factory = env_factory
        child_keys = key.spawn(num_envs)
        self.envs: List[SinglePlayerEnv] = []
        for i in range(num_envs):
            engine_rng, env_rng = child_keys[i].spawn(2)
            engine = env_factory.create(engine_rng)
            self.envs.append(SinglePlayerEnv(engine, opponent.slice(i), rng=env_rng))

        self._stepper = batch_stepper

        # Lazy-init buffers (generic path)
        self._out_obs = None
        self._opp_obs = None
        self._out_masks = None
        self._opp_masks = None
        self._rewards = None
        self._terminated = None
        self._truncated = None

    def _init_buffers(self):
        N = self.num_envs
        sample_obs = self.envs[0].engine.encode(0)
        num_actions = self.envs[0].engine.num_actions
        self._out_obs = _alloc_obs_buf(sample_obs, N)
        self._opp_obs = _alloc_obs_buf(sample_obs, N)
        self._out_masks = np.zeros((N, num_actions), dtype=bool)
        self._opp_masks = np.zeros((N, num_actions), dtype=bool)
        self._rewards = np.zeros(N, dtype=np.float32)
        self._terminated = np.zeros(N, dtype=bool)
        self._truncated = np.zeros(N, dtype=bool)

    def reset(self):
        N = self.num_envs

        if self._stepper is not None:
            # BatchStepper path: reset envs, then batch-encode in JIT
            seats = np.zeros(N, dtype=np.int8)
            for i, env in enumerate(self.envs):
                env.reset()
                seats[i] = env._seat
            self._stepper.init(self.envs, seats)
            self._stepper.encode_initial()
            return self._stepper.obs, self._stepper.masks

        # Generic path. Reset envs FIRST: _init_buffers() samples a template
        # obs via engine.encode(), which requires a valid (post-reset) engine
        # state. Engines whose state is None before reset (r_eco, ttr) would
        # otherwise raise in encode().
        for env in self.envs:
            env.reset()
        if self._out_obs is None:
            self._init_buffers()
        for i, env in enumerate(self.envs):
            env.engine.encode_into(env._seat, self._out_obs, i)
            self._out_masks[i] = env.engine.legal_actions()
        return self._out_obs, self._out_masks

    def step(self, actions: np.ndarray):
        if self._stepper is not None:
            return self._step_stepper(actions)
        return self._step_generic(actions)

    # ── BatchStepper path (game-agnostic) ─────────────────────────────

    def _step_stepper(self, actions: np.ndarray):
        stepper = self._stepper

        # 1. Validate + step agents + find opponents (one JIT call)
        n_p = stepper.step_and_find_opponents(actions)

        # 2. Opponent loop — only escapes to Python for NN inference
        while n_p > 0:
            obs_slice, mask_slice, indices = stepper.get_opponent_slice(n_p)
            opp_actions = self._opponent.batch_action(obs_slice, mask_slice, indices)
            n_p = stepper.apply_and_find_more(n_p, opp_actions)

        # 3. Collect results + metrics + reset done envs + encode (all in JIT)
        rewards, done_f32, done_indices, n_done = stepper.collect_and_encode()

        # 4. Post-reset opponent handling
        #    Envs that reset with opponent going first need opponent turns.
        if n_done > 0:
            n_p = stepper.handle_post_reset_opponents()
            while n_p > 0:
                obs_slice, mask_slice, indices = stepper.get_opponent_slice(n_p)
                opp_actions = self._opponent.batch_action(obs_slice, mask_slice, indices)
                n_p = stepper.apply_and_find_more(n_p, opp_actions)
            stepper.re_encode_done(n_done)

        # Return format: (obs, masks, rewards, terminated_bool, truncated_bool, infos)
        # infos is StepInfo object (pre-allocated arrays, not list of dicts)
        terminated = stepper.info.is_done
        truncated = np.zeros(self.num_envs, dtype=bool)
        return (stepper.obs, stepper.masks,
                rewards, terminated, truncated, stepper.info)

    # ── Generic per-env path (fallback) ────────────────────────────────

    def _step_generic(self, actions: np.ndarray):
        assert len(actions) == self.num_envs
        N = self.num_envs

        if self._out_obs is None:
            self._init_buffers()

        for i in range(N):
            mask = self.envs[i].engine.legal_actions()
            a = int(actions[i])
            if not mask[a]:
                legal = np.where(mask)[0]
                actions[i] = int(legal[0]) if len(legal) > 0 else 0

        for i in range(N):
            env = self.envs[i]
            rewards = env.engine.step(int(actions[i]))
            env._accumulated_reward += rewards[env._seat]

        while True:
            pending = []
            for i in range(N):
                env = self.envs[i]
                if not env.engine.done and env.engine.current_player != env._seat:
                    pending.append(i)
            if not pending:
                break
            n_p = len(pending)
            for j, i in enumerate(pending):
                eng = self.envs[i].engine
                eng.encode_into(eng.current_player, self._opp_obs, j)
                self._opp_masks[j] = eng.legal_actions()
            opp_obs_slice = _slice_obs(self._opp_obs, n_p)
            opp_actions = self._opponent.batch_action(
                opp_obs_slice, self._opp_masks[:n_p], pending)
            for j, i in enumerate(pending):
                env = self.envs[i]
                rewards = env.engine.step(int(opp_actions[j]))
                env._accumulated_reward += rewards[env._seat]

        self._rewards[:] = 0.0
        self._terminated[:] = False
        self._truncated[:] = False
        infos: list = [{} for _ in range(N)]

        for i in range(N):
            env = self.envs[i]
            self._rewards[i] = env._accumulated_reward
            env._accumulated_reward = 0.0
            if env.engine.done:
                self._terminated[i] = True
                info: dict = {"agent_seat": env._seat}
                try:
                    info["final_scores"] = env.engine.compute_scores()
                except NotImplementedError:
                    pass
                try:
                    info["game_metrics"] = env.engine.game_metrics(env._seat)
                except (NotImplementedError, AttributeError):
                    pass
                info["terminal_obs"] = env.engine.encode(env._seat)
                infos[i] = info
                env.reset()
            env.engine.encode_into(env._seat, self._out_obs, i)
            self._out_masks[i] = env.engine.legal_actions()

        return (self._out_obs, self._out_masks,
                self._rewards, self._terminated, self._truncated, infos)

    def close(self):
        pass
