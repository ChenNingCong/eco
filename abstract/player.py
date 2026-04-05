"""
Abstract player interfaces for vectorized environments.

SlicedPlayer  — per-env opponent: action(obs, mask) -> int
BasePlayer    — batched opponent: batch_action, reset, slice
"""

from abc import ABC, abstractmethod
import numpy as np


class SlicedPlayer(ABC):
    """Per-env opponent interface. Used by SinglePlayerEnv."""

    @abstractmethod
    def action(self, obs, mask) -> int:
        ...


class BasePlayer(ABC):
    """
    Batched opponent interface used by VecSinglePlayerEnv.

    - batch_action(obs, mask, idxs): batched inference for a subset of envs
    - reset(env_indices): reset internal state for given envs (or all)
    - slice(i): returns a SlicedPlayer for a single env
    """

    @abstractmethod
    def batch_action(self, obs_batch, mask_batch, idxs: list) -> np.ndarray:
        ...

    @abstractmethod
    def reset(self, env_indices: list[int] | None = None) -> None:
        ...

    @abstractmethod
    def slice(self, env_idx: int) -> SlicedPlayer:
        ...


class RandomPlayer(BasePlayer, SlicedPlayer):
    """Random opponent. Stateless — picks uniformly from legal actions."""

    def batch_action(self, obs_batch, mask_batch, idxs=None) -> np.ndarray:
        return np.array([
            int(np.random.choice(np.where(m)[0])) for m in mask_batch
        ], dtype=np.int32)

    def reset(self, env_indices=None) -> None:
        pass

    def slice(self, env_idx: int) -> "RandomPlayer":
        return self

    def action(self, obs, mask) -> int:
        return int(np.random.choice(np.where(mask)[0]))


class OffsetPlayer(BasePlayer):
    """Maps local env indices [0,K) → global [start, start+K) in a parent player."""

    def __init__(self, parent: BasePlayer, start_idx: int, num_envs: int):
        self.parent = parent
        self.start_idx = start_idx
        self.num_envs = num_envs

    def batch_action(self, obs_batch, mask_batch, idxs: list) -> np.ndarray:
        return self.parent.batch_action(obs_batch, mask_batch,
                                         [i + self.start_idx for i in idxs])

    def reset(self, env_indices=None):
        if env_indices is None:
            self.parent.reset(list(range(self.start_idx, self.start_idx + self.num_envs)))
        else:
            self.parent.reset([i + self.start_idx for i in env_indices])

    def slice(self, env_idx: int) -> SlicedPlayer:
        return self.parent.slice(env_idx + self.start_idx)
