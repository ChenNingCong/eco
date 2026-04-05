"""
Vectorized single-player environment — runs N SinglePlayerEnvs in lockstep.

Uses generator-based batched opponent evaluation: all N envs yield their
pending opponent observations, the caller evaluates in one batched NN call,
and sends actions back. This turns O(N) NN calls into O(rounds) batched calls.

NOT customizable — works with any SinglePlayerEnv / BaseGameEngine.

Interface:
    obs, masks = vec.reset()
    obs, masks, rewards, terminated, truncated, infos = vec.step(actions)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List

from .game import BaseGameEngine
from .single_player_env import SinglePlayerEnv
from .player import BasePlayer
from .key import Key


def _stack_obs(obs_list: list):
    """Stack a list of obs pytrees (NamedTuples) into a batched obs."""
    cls = type(obs_list[0])
    return cls(**{
        field: np.stack([getattr(o, field) for o in obs_list], axis=0)
        for field in cls._fields
    })


class EnvFactory(ABC):
    """
    Factory that creates game engines for each env slot.

    The factory holds only config (game parameters, wrapper settings, etc.).
    RNG instances are passed in by the caller.

    Subclass and implement `create(rng)`.
    """

    @abstractmethod
    def create(self, rng: np.random.Generator) -> BaseGameEngine:
        """
        Create a game engine with the provided RNG.

        Parameters
        ----------
        rng : np.random.Generator for the engine's randomness
        """
        ...


class VecSinglePlayerEnv:
    """
    Synchronous vectorised single-player environment.

    Parameters
    ----------
    num_envs    : int
    opponent    : BasePlayer — batched opponent with batch_action/slice/reset
    env_factory : EnvFactory — creates SinglePlayerEnv per slot (config only)
    key         : Key — master key; converted to per-env RNGs then discarded
    """

    def __init__(
        self,
        num_envs: int,
        opponent: BasePlayer,
        env_factory: EnvFactory,
        key: Key,
    ):
        self.num_envs = num_envs
        self._opponent = opponent
        self._factory = env_factory
        # Spawn per-env keys, then split each into engine + env RNGs
        child_keys = key.spawn(num_envs)
        self.envs: List[SinglePlayerEnv] = []
        for i in range(num_envs):
            engine_rng, env_rng = child_keys[i].spawn(2)
            engine = env_factory.create(engine_rng)
            self.envs.append(SinglePlayerEnv(engine, opponent.slice(i), rng=env_rng))

    def reset(self):
        obs_list, masks = [], []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
            masks.append(env.legal_actions())
        return _stack_obs(obs_list), np.stack(masks, axis=0)

    def step(self, actions: np.ndarray):
        assert len(actions) == self.num_envs
        N = self.num_envs

        # Validate actions
        for i, env in enumerate(self.envs):
            mask = env.legal_actions()
            a = int(actions[i])
            if not mask[a]:
                legal = np.where(mask)[0]
                actions[i] = int(legal[0]) if len(legal) > 0 else 0

        # Start generators
        gens = [env.step_gen(int(a)) for env, a in zip(self.envs, actions)]
        results = [None] * N

        # Prime generators
        pending = {}
        for i in range(N):
            try:
                opp_obs, opp_mask = next(gens[i])
                pending[i] = (opp_obs, opp_mask)
            except StopIteration as e:
                results[i] = e.value

        # Drive generators with batched opponent inference
        while pending:
            idxs = list(pending.keys())
            obs_list = [pending[i][0] for i in idxs]
            mask_list = [pending[i][1] for i in idxs]

            obs_batch = _stack_obs(obs_list)
            mask_batch = np.stack(mask_list, axis=0)
            opp_actions = self._opponent.batch_action(obs_batch, mask_batch, idxs)

            new_pending = {}
            for j, i in enumerate(idxs):
                try:
                    opp_obs, opp_mask = gens[i].send(int(opp_actions[j]))
                    new_pending[i] = (opp_obs, opp_mask)
                except StopIteration as e:
                    results[i] = e.value
            pending = new_pending

        # Auto-reset terminated envs
        obs_list_out = []
        masks_out = []
        rewards = np.zeros(N, dtype=np.float32)
        terminated = np.zeros(N, dtype=bool)
        truncated = np.zeros(N, dtype=bool)
        infos = []

        for i in range(N):
            obs, rew, term, trunc, info = results[i]
            if term:
                info["terminal_obs"] = obs
                obs, _ = self.envs[i].reset()
            obs_list_out.append(obs)
            masks_out.append(self.envs[i].legal_actions())
            rewards[i] = float(rew)
            terminated[i] = term
            infos.append(info)

        return (_stack_obs(obs_list_out), np.stack(masks_out, axis=0),
                rewards, terminated, truncated, infos)

    def close(self):
        pass
