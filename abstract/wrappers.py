"""
Game wrappers — gymnasium-style transforms applied to a BaseGameEngine.

Wrappers compose: RewardShaping(engine).
Each wrapper IS-A BaseGameEngine, so downstream code doesn't care.
The wrapper delegates key/RNG to the inner engine — no separate key needed.
"""

import numpy as np
from .game import BaseGameEngine, Obs
from .key import Key


class GameWrapper(BaseGameEngine[Obs]):
    """
    Base wrapper that delegates everything to the inner engine.
    Subclass and override specific methods to add behavior.
    """

    def __init__(self, engine: BaseGameEngine[Obs]):
        # Don't call super().__init__ — we delegate key to the inner engine
        self.engine = engine

    @property
    def key(self) -> Key:
        return self.engine.key

    @key.setter
    def key(self, value: Key):
        self.engine.key = value

    def reset(self) -> None:
        self.engine.reset()

    def step(self, action: int) -> tuple[float, ...]:
        return self.engine.step(action)

    def legal_actions(self) -> np.ndarray:
        return self.engine.legal_actions()

    def encode(self, player_id: int) -> Obs:
        return self.engine.encode(player_id)

    @property
    def current_player(self) -> int:
        return self.engine.current_player

    @property
    def done(self) -> bool:
        return self.engine.done

    @property
    def num_players(self) -> int:
        return self.engine.num_players

    @property
    def num_actions(self) -> int:
        return self.engine.num_actions


class RewardShaping(GameWrapper[Obs]):
    """
    Applies per-step reward shaping: r_shaped = scale * (my_r - penalty * max_opp_r).
    """

    def __init__(self, engine: BaseGameEngine[Obs], scale: float = 1.0,
                 opponent_penalty: float = 0.5):
        super().__init__(engine)
        self.scale = scale
        self.opponent_penalty = opponent_penalty

    def step(self, action: int) -> tuple[float, ...]:
        rewards = self.engine.step(action)
        n = self.num_players
        shaped = []
        for p in range(n):
            opp_best = max(rewards[i] for i in range(n) if i != p)
            shaped.append(self.scale * (rewards[p] - self.opponent_penalty * opp_best))
        return tuple(shaped)
