"""
Lost Cities actor-critic agent and game-specific training config.

Architecture: 2-layer encoder + parallel LSTM, concat, 2-layer trunks.
Sized for LC: 301-dim obs, 600 actions (smaller game than TTR).
"""
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from abstract import (
    BaseAgent, LSTMState, PPOConfig,
    CategoricalMasked, layer_init,
)
from .engine import (LCObs, float_dim, NUM_ACTIONS, NUM_DECOMPOSED_ACTIONS, NUM_3PHASE_ACTIONS,
                      NUM_CARD_IDS, NUM_ACTION_TYPES, NUM_DRAW_SOURCES)


@dataclass
class LCArgs(PPOConfig):
    """Lost Cities game-specific arguments, extends PPOConfig."""
    opponent_mode: Literal["self_play", "random", "heuristic"] = "self_play"
    """opponent policy: self_play, random, or heuristic (rule-based fixed opponent)."""
    new_color_penalty: int = 20
    """penalty for starting a new color expedition (default 20, ablations: 15, 10)"""
    score_diff_reward: bool = False
    """use (score_diff / 30) as terminal reward instead of +1/-1"""
    zero_one_reward: bool = False
    """use [0,1] reward (win=1, lose=0) instead of +1/-1"""
    dense_reward: bool = False
    """per-step reward (score delta each move) instead of terminal-only. Combine with
    score_diff_reward for the dense zero-sum (own-opp) signal: r=(Δown-Δopp)/30 each step."""
    max_lanes: int = 5
    """maximum number of expedition lanes a player can open (default 5 = no limit)"""
    decompose_actions: bool = False
    """decompose flat 600-action space into sequential play(100)+draw(6)=106 actions"""
    three_phase: bool = False
    """3-phase decomposition: select_card(50)+action_type(2)+draw(6)=58 actions"""
    blind_draw: bool = False
    """decompose actions but do NOT feed played_card/played_type to draw phase (ablation)"""
    no_lstm: bool = False
    """pure feedforward agent (no LSTM history), uses 2-layer middle MLP instead"""
    hidden_dim: int = 256
    """hidden dimension for encoder/trunk MLPs (default 256)"""
    mlp_depth: int = 2
    """number of layers per MLP block (encoder/middle/actor/critic); 2 = original arch"""
    mask_as_input: bool = False
    """feed action mask as additional input feature to the encoder"""
    product_actions: bool = False
    """use product (factored) action heads: card(50)×type(2)×draw(6) combined via additive log-space"""
    max_discard_draws: int = 0
    """max total discard draws before masking; 0=unlimited"""
    pretrained: Optional[str] = None
    """path to pretrained PPO checkpoint (e.g. from behavioral cloning)"""


class LCAgent(BaseAgent):
    """
    Lost Cities actor-critic: 2-layer encoder + parallel LSTM, concat, 2-layer trunks.

    Architecture: flat_enc(float_d [+ mask_d] → H) [+ phase_emb]
                    → fusion → LSTM(fusion→lstm_hidden)
                    → concat(fusion, lstm_hidden)
                      ├→ actor_trunk(2 layers) → head(num_actions)
                      └→ critic_trunk(2 layers) → head(1)

    Flags:
      decompose_actions: split 600 flat actions into play(100)+draw(6)=106,
                         adds phase embedding
      mask_as_input: feed action mask as additional encoder input feature
    """
    HIDDEN = 256
    PHASE_EMB_DIM = 32

    def __init__(self, lstm_hidden: int = 128, lstm_layers: int = 1,
                 decompose_actions: bool = False, three_phase: bool = False,
                 blind_draw: bool = False,
                 mask_as_input: bool = False, no_lstm: bool = False,
                 hidden_dim: int = 256, product_actions: bool = False,
                 mlp_depth: int = 2):
        super().__init__()
        H = hidden_dim
        self.lstm_hidden = lstm_hidden if not no_lstm else 1
        self.lstm_layers = lstm_layers
        self.decompose_actions = decompose_actions
        self.three_phase = three_phase
        self.blind_draw = blind_draw
        self.mask_as_input = mask_as_input
        self.no_lstm = no_lstm
        self.product_actions = product_actions

        # Determine action count
        if three_phase:
            self._num_actions = NUM_3PHASE_ACTIONS  # 58
        elif decompose_actions:
            self._num_actions = NUM_DECOMPOSED_ACTIONS  # 106
        else:
            self._num_actions = NUM_ACTIONS  # 600

        # Encoder input: float features + optional action mask
        uses_played_info = (decompose_actions and not blind_draw) or three_phase
        enc_in = float_dim(decompose_actions=(decompose_actions and not blind_draw),
                           three_phase=three_phase)
        if mask_as_input:
            enc_in += self._num_actions

        # MLP block builder: `mlp_depth` hidden layers of width H (depth=2 = original arch)
        def _mlp(in_dim):
            layers = [layer_init(nn.Linear(in_dim, H)), nn.LayerNorm(H), nn.ReLU()]
            for _ in range(mlp_depth - 1):
                layers += [layer_init(nn.Linear(H, H)), nn.LayerNorm(H), nn.ReLU()]
            return nn.Sequential(*layers)

        # encoder for float features (+ optional mask)
        self.flat_enc = _mlp(enc_in)

        # Phase embedding (used for decompose_actions or three_phase)
        if three_phase:
            self.phase_emb = nn.Embedding(3, self.PHASE_EMB_DIM)
            fusion_in = H + self.PHASE_EMB_DIM
        elif decompose_actions:
            self.phase_emb = nn.Embedding(2, self.PHASE_EMB_DIM)
            fusion_in = H + self.PHASE_EMB_DIM
        else:
            fusion_in = H

        if no_lstm:
            # Dummy LSTM (hidden=1) to satisfy trainer's state management
            self.lstm = nn.LSTM(fusion_in, 1, num_layers=lstm_layers)
            # middle MLP replaces LSTM for deeper representation
            self.middle_mlp = _mlp(fusion_in)
            trunk_in = H
        else:
            # LSTM parallel to FF path
            self.lstm = nn.LSTM(fusion_in, lstm_hidden, num_layers=lstm_layers)
            for name, param in self.lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
            trunk_in = fusion_in + lstm_hidden
        self.actor_trunk = _mlp(trunk_in)
        self.critic_trunk = _mlp(trunk_in)
        if product_actions:
            # Factored action heads: logits = card[c] + type[t] + draw[d]
            self.card_head = layer_init(nn.Linear(H, NUM_CARD_IDS), std=0.01)      # 50
            self.type_head = layer_init(nn.Linear(H, NUM_ACTION_TYPES), std=0.01)   # 2
            self.draw_head = layer_init(nn.Linear(H, NUM_DRAW_SOURCES), std=0.01)   # 6
        else:
            self.actor_head = layer_init(nn.Linear(H, self._num_actions), std=0.01)
        self.critic_head = layer_init(nn.Linear(H, 1), std=1.0)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def _shared_encode(self, obs: LCObs, action_mask=None) -> torch.Tensor:
        parts = [
            obs.hand, obs.own_expeditions, obs.opp_expeditions,
            obs.discard_top1, obs.discard_top2, obs.discard_top3,
            obs.deck_size, obs.scores,
        ]
        if self.three_phase:
            parts.extend([obs.played_card, obs.played_type])
        elif self.decompose_actions and not self.blind_draw:
            parts.extend([obs.played_card, obs.played_type])
        if self.mask_as_input and action_mask is not None:
            parts.append(action_mask.float())
        flat = torch.cat(parts, dim=-1)
        enc = self.flat_enc(flat)  # (B, H)

        if self.three_phase or self.decompose_actions:
            phase_repr = self.phase_emb(obs.phase.squeeze(-1))  # (B, PHASE_EMB_DIM)
            enc = torch.cat([enc, phase_repr], dim=-1)  # (B, H + PHASE_EMB_DIM)

        return enc

    def get_states(self, obs: LCObs, lstm_state: LSTMState, done: torch.Tensor,
                   action_mask=None):
        shared = self._shared_encode(obs, action_mask)

        if self.no_lstm:
            # Pure feedforward: middle MLP replaces LSTM
            batch_size = lstm_state.h.shape[1]
            dummy_in = shared[:batch_size].unsqueeze(0)
            _, (h, c) = self.lstm(dummy_in, (lstm_state.h, lstm_state.c))
            return self.middle_mlp(shared), LSTMState(h=h, c=c)

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

    def get_value(self, obs: LCObs, lstm_state: LSTMState, done: torch.Tensor,
                  action_mask=None):
        hidden, _ = self.get_states(obs, lstm_state, done, action_mask)
        return self.critic_head(self.critic_trunk(hidden))

    def _compute_logits(self, actor_h: torch.Tensor) -> torch.Tensor:
        if self.product_actions:
            card_logits = self.card_head(actor_h)    # (B, 50)
            type_logits = self.type_head(actor_h)    # (B, 2)
            draw_logits = self.draw_head(actor_h)    # (B, 6)
            # Outer sum: logits[c,t,d] = card[c] + type[t] + draw[d]
            # Reshape and broadcast to (B, 50, 2, 6) then flatten to (B, 600)
            logits = (card_logits[:, :, None, None]
                      + type_logits[:, None, :, None]
                      + draw_logits[:, None, None, :])
            return logits.reshape(actor_h.shape[0], -1)
        return self.actor_head(actor_h)

    def get_action_and_value(self, obs: LCObs, action_mask,
                             lstm_state: LSTMState, done: torch.Tensor,
                             action=None):
        hidden, new_lstm_state = self.get_states(obs, lstm_state, done, action_mask)
        actor_h = self.actor_trunk(hidden)
        logits = self._compute_logits(actor_h)
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
