"""
Abstract game engine interface for multi-player turn-based games.

A game engine manages game state and exposes:
  - reset(): start a new game (uses internal mutable RNG)
  - step(action): current player performs action, returns per-player rewards
  - legal_actions(): bool mask of legal actions for the current player
  - current_player: which player acts next
  - done: whether the game is over
  - encode(player_id): observation from a specific player's perspective

The observation type (Obs) is game-specific — typically a NamedTuple of numpy
arrays (a pytree). The framework is agnostic to its structure.
"""

import random
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import numpy as np

# Obs is the game-specific observation type (e.g. a NamedTuple pytree)
Obs = TypeVar("Obs")


class BaseGameEngine(ABC, Generic[Obs]):
    """
    Abstract multi-player turn-based game engine.

    Subclasses must implement the game logic. The engine is stateful:
    it holds the current game state internally. RNG is a mutable
    random.Random instance passed at construction time.
    """

    def __init__(self, rng: random.Random):
        self.rng: random.Random = rng

    @abstractmethod
    def reset(self) -> None:
        """Reset to a new game. Uses self.rng for randomness."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[float, ...]:
        """
        Current player performs `action`. Mutates internal state.
        Check `done` property after to see if game ended.

        Returns
        -------
        rewards : tuple of float (length num_players) — per-player reward
        """
        ...

    @abstractmethod
    def legal_actions(self) -> np.ndarray:
        """Bool mask (num_actions,) of legal actions for the current player."""
        ...

    @abstractmethod
    def encode(self, player_id: int) -> Obs:
        """
        Observation from `player_id`'s perspective.

        The returned Obs should be a pytree (e.g. NamedTuple of numpy arrays).
        Player-indexed data should be rotated so that `player_id` is always
        at index 0 (the network doesn't need to learn seat-awareness).
        """
        ...

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Which player acts next."""
        ...

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the game is over."""
        ...

    @property
    @abstractmethod
    def num_players(self) -> int:
        ...

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Size of the action space."""
        ...
