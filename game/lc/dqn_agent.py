"""
Lost Cities Q-network and DQN training config.

Architecture: same encoder as LCAgent but outputs Q-values instead of policy+value.
"""
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from abstract.dqn import DQNConfig, BaseQNetwork
from abstract.ppo_lstm import layer_init
from .engine import float_dim, NUM_ACTIONS


@dataclass
class LCDQNArgs(DQNConfig):
    """Lost Cities DQN arguments."""
    opponent_mode: Literal["self_play", "random"] = "self_play"
    new_color_penalty: int = 20
    score_diff_reward: bool = False
    zero_one_reward: bool = False
    max_lanes: int = 5
    max_discard_draws: int = 50
    """max total discard draws (both players) before discard draws are masked; 0=unlimited"""
    raw_score_reward: bool = False
    """reward = own score / 30 (maximize raw score, ignore opponent)"""
    dense_reward: bool = False
    """per-step reward = delta in current player's score / 30"""


class LCQNetwork(BaseQNetwork):
    """
    Q-network for Lost Cities (flat 600 action space).

    Architecture: 2-layer encoder → 2-layer trunk → Q-values (600)
    Same depth as LCAgent but no LSTM, no actor/critic split.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        obs_dim = float_dim()  # 303 (base obs, no decompose)
        H = hidden_dim
        self._num_actions = NUM_ACTIONS  # 600

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, H)), nn.LayerNorm(H), nn.ReLU(),
            layer_init(nn.Linear(H, H)),       nn.LayerNorm(H), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(H, H)), nn.LayerNorm(H), nn.ReLU(),
            layer_init(nn.Linear(H, H)), nn.LayerNorm(H), nn.ReLU(),
        )
        self.head = layer_init(nn.Linear(H, self._num_actions), std=0.01)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        # obs_flat is (B, total_obs_dim) — all fields concatenated as float32
        # We only need the first float_dim() features (the rest are phase/played_card etc
        # which are always 0 for flat action space)
        enc = self.encoder(obs_flat[:, :float_dim()])
        return self.head(self.trunk(enc))
