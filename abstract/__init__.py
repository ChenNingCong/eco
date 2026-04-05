"""
Abstract game framework for multi-player turn-based games.

Layers (bottom to top):
  1. BaseGameEngine  — game logic (customizable)
  2. GameWrapper     — reward shaping etc. (customizable, composable)
  3. SinglePlayerEnv — wraps multi-player → single-agent (not customizable)
  4. VecSinglePlayerEnv — vectorized (not customizable)
  5. MultiProcessVecEnv — multiprocessing (not customizable)

Players:
  SlicedPlayer / BasePlayer / RandomPlayer / OffsetPlayer
"""

from .game import BaseGameEngine
from .wrappers import GameWrapper, RewardShaping
from .single_player_env import SinglePlayerEnv
from .vec_env import VecSinglePlayerEnv, EnvFactory
from .player import SlicedPlayer, BasePlayer, RandomPlayer, OffsetPlayer
from .key import Key, key_from_seed
from .mp_vec_env import MultiProcessVecEnv
from .ppo_lstm import (
    PPOConfig, BaseAgent, LSTMState, make_lstm_state,
    LSTMBatchedPlayer, LSTMSlicedPlayer, PPOLSTMTrainer,
    CategoricalMasked, layer_init,
    obs_to_tensor, obs_unsqueeze, alloc_obs_buffer,
)
