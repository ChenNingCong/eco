"""
R-Öko actor-critic agent and game-specific training config.
"""
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from abstract import (
    BaseAgent, LSTMState, PPOConfig,
    CategoricalMasked, layer_init,
)
from .engine import RÖkoObs, float_dim, NUM_ACTIONS


@dataclass
class EcoArgs(PPOConfig):
    """R-Öko game-specific arguments, extends PPOConfig."""
    num_players: int = 2
    """number of players per game"""
    reward_shaping_scale: float = 1.0
    """scaling factor for per-action intermediate rewards (0 = disabled)."""
    opponent_penalty: float = 0.5
    """penalty for opponent scoring in shaping reward: r = my_r - penalty * max(opp_r)."""
    opponent_mode: Literal["self_play", "random"] = "self_play"
    """opponent policy: self_play or random."""


class EcoAgent(BaseAgent):
    """
    R-öko actor-critic: FF encoder + parallel LSTM, concat, 2-layer trunks.
    Pure/stateless: hidden state passed in/out.

    Architecture: shared_encode → fusion(320)
                                    → LSTM(320→lstm_hidden)
                                    → concat(fusion, lstm_out) = 320 + lstm_hidden
                                      ├→ actor_trunk(2 layers) → head
                                      └→ critic_trunk(2 layers) → head
    """
    EMB_DIM = 32
    HIDDEN  = 256

    def __init__(self, num_players: int = 2,
                 lstm_hidden: int = 128, lstm_layers: int = 1):
        super().__init__()
        E, H = self.EMB_DIM, self.HIDDEN
        self.num_players = num_players
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        float_d = float_dim(num_players)

        self.player_emb = nn.Embedding(num_players + 1, E)   # tokens 1..num_players
        self.phase_emb  = nn.Embedding(2, E)                 # 0=play, 1=discard

        # Shared encoder: float features → H
        self.flat_enc = nn.Sequential(
            layer_init(nn.Linear(float_d, H)), nn.LayerNorm(H), nn.ReLU(),
            layer_init(nn.Linear(H, H)),       nn.LayerNorm(H), nn.ReLU(),
        )
        fusion_in = H + 2 * E

        # LSTM takes encoder output, produces additional recurrent features
        self.lstm = nn.LSTM(fusion_in, lstm_hidden, num_layers=lstm_layers)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # 2-layer trunks take concat(fusion, lstm_out) — full width
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
        self.actor_head  = layer_init(nn.Linear(H, NUM_ACTIONS), std=0.01)
        self.critic_head = layer_init(nn.Linear(H, 1), std=1.0)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def _shared_encode(self, obs: RÖkoObs) -> torch.Tensor:
        # Embed discrete tokens
        player_repr = self.player_emb(obs.current_player.squeeze(-1))   # (B, E)
        phase_repr  = self.phase_emb(obs.phase.squeeze(-1))             # (B, E)

        # Encode all float features
        flat = torch.cat([
            obs.hands, obs.recycling_side, obs.waste_side,
            obs.factory_stacks, obs.collected,
            obs.penalty_pile, obs.scores, obs.draw_pile_size,
            obs.draw_pile_comp,
        ], dim=-1)
        flat_repr = self.flat_enc(flat)                                  # (B, H)

        return torch.cat([flat_repr, player_repr, phase_repr], dim=-1)   # (B, fusion_in)

    def get_states(self, obs: RÖkoObs, lstm_state: LSTMState, done: torch.Tensor):
        """
        Encode obs through shared encoder + LSTM, resetting hidden on done.
        Returns (concat(shared, lstm_out) of shape (T*B, fusion_in+lstm_hidden), new_lstm_state).
        """
        shared = self._shared_encode(obs)

        # LSTM: process sequentially, resetting state on episode boundaries
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

        # Concat FF features + LSTM features — trunks see both
        combined = torch.cat([shared, lstm_out], dim=-1)
        return combined, LSTMState(h=h, c=c)

    def get_value(self, obs: RÖkoObs, lstm_state: LSTMState, done: torch.Tensor):
        hidden, _ = self.get_states(obs, lstm_state, done)
        return self.critic_head(self.critic_trunk(hidden))

    def get_action_and_value(self, obs: RÖkoObs, action_mask,
                             lstm_state: LSTMState, done: torch.Tensor,
                             action=None):
        hidden, new_lstm_state = self.get_states(obs, lstm_state, done)
        actor_h = self.actor_trunk(hidden)
        logits = self.actor_head(actor_h)
        probs  = CategoricalMasked(logits=logits, masks=action_mask)
        if action is None:
            action = probs.sample()
            # Safety: force legal if sampling produced illegal action
            illegal = ~action_mask.gather(1, action.unsqueeze(1)).squeeze(1)
            if illegal.any():
                n = illegal.sum().item()
                print(f"[WARN] Hard mask enforcement triggered for {n}/{len(action)} actions")
                masked_logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=logits.device))
                fallback = masked_logits.argmax(dim=1)
                action = torch.where(illegal, fallback, action)
        critic_h = self.critic_trunk(hidden)
        return action, probs.log_prob(action), probs.entropy(), self.critic_head(critic_h), new_lstm_state
