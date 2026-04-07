"""EnvFactory for Lost Cities."""

import numpy as np

from abstract import EnvFactory
from .engine import LCEngine


class LCEnvFactory(EnvFactory):
    """Creates LCEngine instances with configurable penalty and reward mode."""

    def __init__(self, new_color_penalty: int = 20, score_diff_reward: bool = False,
                 zero_one_reward: bool = False,
                 max_lanes: int = 5, decompose_actions: bool = False,
                 three_phase: bool = False, max_discard_draws: int = 0,
                 raw_score_reward: bool = False, dense_reward: bool = False):
        self.new_color_penalty = new_color_penalty
        self.score_diff_reward = score_diff_reward
        self.zero_one_reward = zero_one_reward
        self.raw_score_reward = raw_score_reward
        self.dense_reward = dense_reward
        self.max_lanes = max_lanes
        self.decompose_actions = decompose_actions
        self.three_phase = three_phase
        self.max_discard_draws = max_discard_draws

    def create(self, rng: np.random.Generator) -> LCEngine:
        return LCEngine(rng=rng, new_color_penalty=self.new_color_penalty,
                         score_diff_reward=self.score_diff_reward,
                         zero_one_reward=self.zero_one_reward,
                         raw_score_reward=self.raw_score_reward,
                         max_lanes=self.max_lanes,
                         decompose_actions=self.decompose_actions,
                         three_phase=self.three_phase,
                         max_discard_draws=self.max_discard_draws,
                         dense_reward=self.dense_reward)
