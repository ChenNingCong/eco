"""EnvFactory for Ticket to Ride."""

import numpy as np

from abstract import EnvFactory
from .engine import TTREngine


class TTREnvFactory(EnvFactory):
    """Creates TTREngine instances. Config: num_players."""

    def __init__(self, num_players: int = 2):
        self.num_players = num_players

    def create(self, rng: np.random.Generator) -> TTREngine:
        return TTREngine(rng=rng, num_players=self.num_players)
