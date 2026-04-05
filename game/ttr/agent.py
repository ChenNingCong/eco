"""
Ticket to Ride actor-critic agent and game-specific training config.

Architecture: larger than R-Öko agent (bigger observation, more strategic depth).
  - 3-layer encoder (vs 2 for eco)
  - LSTM parallel to FF path, concat
  - 2-layer actor/critic trunks
"""
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from abstract import (
    BaseAgent, LSTMState, PPOConfig,
    CategoricalMasked, layer_init,
)
from .engine import TTRObs, float_dim, NUM_ACTIONS


@dataclass
class TTRArgs(PPOConfig):
    """TTR game-specific arguments, extends PPOConfig."""
    num_players: int = 2
    """number of players per game (only 2 supported)"""
    opponent_mode: Literal["self_play", "random"] = "self_play"
    """opponent policy: self_play or random."""


class TTRAgent(BaseAgent):
    """
    Ticket to Ride actor-critic: 3-layer encoder + parallel LSTM, concat, 2-layer trunks.

    Architecture: shared_encode → fusion(dim)
                    → LSTM(fusion→lstm_hidden)
                    → concat(fusion, lstm_out)
                      ├→ actor_trunk(2 layers) → head(141)
                      └→ critic_trunk(2 layers) → head(1)
    """
    EMB_DIM = 32
    HIDDEN = 512   # wider than eco (256) — bigger obs/action space

    def __init__(self, num_players: int = 2,
                 lstm_hidden: int = 256, lstm_layers: int = 1):
        super().__init__()
        E, H = self.EMB_DIM, self.HIDDEN
        self.num_players = num_players
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        float_d = float_dim(num_players)

        # Embeddings for game_state (4 vals) and turn_state (4 vals)
        self.game_state_emb = nn.Embedding(4, E)
        self.turn_state_emb = nn.Embedding(4, E)

        # 3-layer encoder for float features
        self.flat_enc = nn.Sequential(
            layer_init(nn.Linear(float_d, H)), nn.LayerNorm(H), nn.ReLU(),
            layer_init(nn.Linear(H, H)),       nn.LayerNorm(H), nn.ReLU(),
            layer_init(nn.Linear(H, H)),       nn.LayerNorm(H), nn.ReLU(),
        )
        fusion_in = H + 2 * E

        # LSTM parallel to FF path
        self.lstm = nn.LSTM(fusion_in, lstm_hidden, num_layers=lstm_layers)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # 2-layer trunks take concat(fusion, lstm_out)
        trunk_in = fusion_in + lstm_hidden
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(trunk_in, H)), nn.LayerNorm(H), nn.ReLU(),
            layer_init(nn.Linear(H, H)),        nn.LayerNorm(H), nn.ReLU(),
        )
        self.critic_trunk = nn.Sequential(
            layer_init(nn.Linear(trunk_in, H)), nn.LayerNorm(H), nn.ReLU(),
            layer_init(nn.Linear(H, H)),        nn.LayerNorm(H), nn.ReLU(),
        )
        self._num_actions = NUM_ACTIONS
        self.actor_head = layer_init(nn.Linear(H, NUM_ACTIONS), std=0.01)
        self.critic_head = layer_init(nn.Linear(H, 1), std=1.0)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def _shared_encode(self, obs: TTRObs) -> torch.Tensor:
        gs_repr = self.game_state_emb(obs.game_state.squeeze(-1))  # (B, E)
        ts_repr = self.turn_state_emb(obs.turn_state.squeeze(-1))  # (B, E)

        flat = torch.cat([
            obs.hands, obs.player_trains, obs.player_points, obs.player_dest_counts,
            obs.visible_cards, obs.deck_size,
            obs.route_ownership, obs.own_dest_status, obs.avail_dest,
            obs.dest_selected,
        ], dim=-1)
        flat_repr = self.flat_enc(flat)  # (B, H)

        return torch.cat([flat_repr, gs_repr, ts_repr], dim=-1)  # (B, fusion_in)

    def get_states(self, obs: TTRObs, lstm_state: LSTMState, done: torch.Tensor):
        shared = self._shared_encode(obs)

        batch_size = lstm_state.h.shape[1]
        lstm_in = shared.reshape((-1, batch_size, shared.shape[-1]))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        h, c = lstm_state.h, lstm_state.c
        for t_h, t_d in zip(lstm_in, done):
            h = (1.0 - t_d).view(1, -1, 1) * h
            c = (1.0 - t_d).view(1, -1, 1) * c
            t_h, (h, c) = self.lstm(t_h.unsqueeze(0), (h, c))
            new_hidden.append(t_h)
        lstm_out = torch.flatten(torch.cat(new_hidden), 0, 1)

        combined = torch.cat([shared, lstm_out], dim=-1)
        return combined, LSTMState(h=h, c=c)

    def get_value(self, obs: TTRObs, lstm_state: LSTMState, done: torch.Tensor):
        hidden, _ = self.get_states(obs, lstm_state, done)
        return self.critic_head(self.critic_trunk(hidden))

    def get_action_and_value(self, obs: TTRObs, action_mask,
                             lstm_state: LSTMState, done: torch.Tensor,
                             action=None):
        hidden, new_lstm_state = self.get_states(obs, lstm_state, done)
        actor_h = self.actor_trunk(hidden)
        logits = self.actor_head(actor_h)
        probs = CategoricalMasked(logits=logits, masks=action_mask)
        if action is None:
            action = probs.sample()
            illegal = ~action_mask.gather(1, action.unsqueeze(1)).squeeze(1)
            if illegal.any():
                n = illegal.sum().item()
                print(f"[WARN] Hard mask enforcement triggered for {n}/{len(action)} actions")
                masked_logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=logits.device))
                fallback = masked_logits.argmax(dim=1)
                action = torch.where(illegal, fallback, action)
        critic_h = self.critic_trunk(hidden)
        return action, probs.log_prob(action), probs.entropy(), self.critic_head(critic_h), new_lstm_state
