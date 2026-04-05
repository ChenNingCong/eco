"""
EnvFactory for R-Öko — config-only factory that creates game engines.
"""

import numpy as np

from abstract import EnvFactory
from .engine import RÖkoEngine


class RÖkoEnvFactory(EnvFactory):
    """
    Creates RÖkoEngine instances.

    Config: num_players.  RNG is provided by the caller.
    """

    def __init__(self, num_players: int = 2):
        self.num_players = num_players

    def create(self, rng: np.random.Generator) -> RÖkoEngine:
        return RÖkoEngine(rng=rng, num_players=self.num_players)
