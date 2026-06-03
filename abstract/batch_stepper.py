"""
Abstract batch stepping interface for game-specific JIT-accelerated env stepping.

When a game engine provides a Numba jitclass (_state), it can also provide a
BatchStepper that moves all env stepping logic into compiled code. The VecEnv
only breaks out of JIT for neural network inference (opponent actions).

Each game implements its own BatchStepper subclass with game-specific @njit
functions — this is the Numba equivalent of virtual dispatch.

Architecture (per step):
  VecEnv.step(actions)
    → stepper.step_and_find_opponents(actions)  # Fused JIT: validate + step + find
    → [opponent NN inference in Python/GPU]
    → stepper.apply_and_find_more(opp_actions)  # JIT: apply + find more
    → stepper.collect_and_encode()              # JIT: rewards + encode agent obs
"""

from abc import ABC, abstractmethod
import numpy as np


class BatchStepper(ABC):
    """Game-specific batch stepping for VecEnv.

    Manages numba.typed.List of jitclass states and pre-allocated numpy buffers.
    All heavy computation runs in @njit functions; Python-level methods are
    thin wrappers that call them.
    """

    @abstractmethod
    def init(self, envs: list, seats: np.ndarray) -> None:
        """Initialize from a list of SinglePlayerEnv instances.
        Build typed list of jitclass states + allocate internal buffers."""
        ...

    @abstractmethod
    def step_and_find_opponents(self, actions: np.ndarray) -> int:
        """Validate actions, step all agents, find pending opponents.
        Writes opponent obs/masks to internal buffers.
        Returns n_pending (0 means no opponent turns needed)."""
        ...

    @abstractmethod
    def get_opponent_slice(self, n_pending: int) -> tuple:
        """Return (obs_namedtuple_slice, mask_slice, env_indices_list).
        The obs/mask slices are views into internal buffers (no copy)."""
        ...

    @abstractmethod
    def apply_and_find_more(self, n_pending: int, opp_actions: np.ndarray) -> int:
        """Apply opponent actions, find more pending opponents.
        Returns new n_pending (0 means opponent loop is done)."""
        ...

    @abstractmethod
    def collect_and_encode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Collect rewards/terminated, encode agent obs+masks.
        Returns (rewards, terminated, done_indices, n_done).
        Agent obs and masks are available via .obs and .masks."""
        ...

    @abstractmethod
    def encode_initial(self) -> None:
        """Encode agent obs+masks for all envs (used after reset)."""
        ...

    @property
    @abstractmethod
    def obs(self):
        """Pre-allocated agent observation buffer (NamedTuple of numpy arrays)."""
        ...

    @property
    @abstractmethod
    def masks(self) -> np.ndarray:
        """Pre-allocated agent legal action masks buffer."""
        ...
