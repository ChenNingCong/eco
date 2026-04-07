"""EnvFactory for Ticket to Ride."""

import numpy as np

from abstract import EnvFactory
from .engine import TTREngine


class TTREnvFactory(EnvFactory):
    """Creates TTREngine instances. Config: num_players + reward shaping."""

    def __init__(self, num_players: int = 2,
                 scale_route_score: float = 1.0, scale_dest_penalty: float = 1.0):
        self.num_players = num_players
        self.scale_route_score = scale_route_score
        self.scale_dest_penalty = scale_dest_penalty

    def create(self, rng: np.random.Generator) -> TTREngine:
        return TTREngine(rng=rng, num_players=self.num_players,
                         scale_route_score=self.scale_route_score,
                         scale_dest_penalty=self.scale_dest_penalty)
