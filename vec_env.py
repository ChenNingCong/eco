"""
Vectorized Hearts environment — single-process, sequential execution.

VecSinglePlayerEnv runs N independent SinglePlayerEnv instances.
Each env has its own agent seat (whoever holds 2♣ in that deal).
Opponents are run internally; every step returned to the caller is an
agent action decision.

Observations: batched PyTreeObs, shape (num_envs, *field_shape) per field.
Masks:        (num_envs, NUM_CARDS) bool — legal cards for each env's agent.
Rewards:      (num_envs,) float32 — accumulated agent reward since last step.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable

from obs_encoder import SinglePlayerEnv, PyTreeObs
from hearts_env import NUM_CARDS, NUM_PLAYERS


def _stack_obs(obs_list: List[PyTreeObs]) -> PyTreeObs:
    return PyTreeObs(**{
        field: np.stack([getattr(o, field) for o in obs_list], axis=0)
        for field in PyTreeObs._fields
    })


class VecSinglePlayerEnv:
    """
    Synchronous vectorised single-player Hearts environment.

    Parameters
    ----------
    num_envs     : int
    opponent_fn  : callable(obs, mask) -> int, optional
        Shared opponent policy for all envs. Defaults to random.
    seeds        : list of int, optional
    """

    def __init__(
        self,
        num_envs: int,
        opponent_fn: Optional[Callable] = None,
        seeds: Optional[List[int]] = None,
        reward_shaping_scale: float = 0.0,
    ):
        self.num_envs = num_envs
        seeds = seeds or [None] * num_envs
        self.envs: List[SinglePlayerEnv] = [
            SinglePlayerEnv(opponent_fn=opponent_fn, seed=s,
                            reward_shaping_scale=reward_shaping_scale)
            for s in seeds
        ]

    def reset(self, seed: Optional[int] = None) -> Tuple[PyTreeObs, np.ndarray]:
        if seed is not None:
            for i, env in enumerate(self.envs):
                env.env.rng = np.random.default_rng(seed + i)

        obs_list, masks = [], []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
            masks.append(env.legal_actions())
        return _stack_obs(obs_list), np.stack(masks, axis=0)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[PyTreeObs, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Returns
        -------
        obs        : PyTreeObs          batched next observation
        masks      : (N, NUM_CARDS)     bool legal action masks
        rewards    : (N,)               float32 accumulated agent reward
        terminated : (N,)               bool
        truncated  : (N,)               bool, always False
        infos      : list of dicts
        """
        assert len(actions) == self.num_envs
        obs_list, masks = [], []
        rewards    = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated  = np.zeros(self.num_envs, dtype=bool)
        infos      = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, rew, term, trunc, info = env.step(int(action))
            if term:
                info["terminal_obs"] = obs
                obs, _ = env.reset()
            obs_list.append(obs)
            masks.append(env.legal_actions())
            rewards[i]    = float(rew)
            terminated[i] = term
            infos.append(info)

        return _stack_obs(obs_list), np.stack(masks, axis=0), rewards, terminated, truncated, infos

    def get_action_masks(self) -> np.ndarray:
        return np.stack([e.legal_actions() for e in self.envs], axis=0)

    def close(self):
        pass


# Backwards-compat alias
VecHeartsEnv = VecSinglePlayerEnv