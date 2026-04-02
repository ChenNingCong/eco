"""
SinglePlayerEcoEnv: single-agent view of R-öko.

The environment runs opponents internally until the agent's seat must act
(play or discard phase).  From the agent's perspective every step() is one
action decision.  Rewards from turns where opponents act are accumulated
and delivered when the agent next acts.

EcoPyTreeObs fields (11 total)
-------------------------------
  current_player   : (1,)                         int32  agent seat token (1-indexed)
  phase            : (1,)                         int32  0=play, 1=discard

  hands            : (num_players*NUM_COLORS*NUM_TYPES,) float32
  recycling_side   : (NUM_COLORS*NUM_TYPES,)       float32
  waste_side       : (NUM_COLORS*NUM_COLORS*NUM_TYPES,)  float32
  factory_stacks   : (NUM_COLORS*stack_size,)      float32  per-slot card values; consumed slots = -1.0, active slots normalised by 5
  collected        : (num_players*NUM_COLORS*2,)   float32  per (player,color): [scores_flag (count>1), value_sum/MAX_SCORE]
  penalty_pile     : (num_players*NUM_COLORS*NUM_TYPES,) float32  penalty pile composition
  scores           : (num_players,)                float32  current scores / MAX_ECO_SCORE (rotated)
  draw_pile_size   : (1,)                         float32  normalised deck size
  draw_pile_comp   : (NUM_COLORS*NUM_TYPES,)       float32  remaining cards by (color,type), normalised

Float dim for 2-player games (stack_size=6): 16+8+32+24+16+16+2+1+8 = 123
(hands=16, recycling=8, waste=32, factory=24, collected=16, penalty_pile=16, scores=2, draw=1, draw_comp=8)
"""

import numpy as np
from typing import Optional, Callable, NamedTuple

from eco_env import (
    EcoEnv, EcoState,
    NUM_COLORS, NUM_TYPES, NUM_ACTIONS,
    MAX_ECO_SCORE, _STACK,
    SINGLES_PER_COLOR, DOUBLES_PER_COLOR,
)

_TOTAL_DECK = NUM_COLORS * (SINGLES_PER_COLOR + DOUBLES_PER_COLOR)  # 88


# ── Observation tuple ─────────────────────────────────────────────────────────

class EcoPyTreeObs(NamedTuple):
    # Embedding index fields (int32)
    current_player   : np.ndarray   # (1,)
    phase            : np.ndarray   # (1,)

    # Continuous fields (float32)
    hands            : np.ndarray   # (num_players * NUM_COLORS * NUM_TYPES,)
    recycling_side   : np.ndarray   # (NUM_COLORS * NUM_TYPES,)
    waste_side       : np.ndarray   # (NUM_COLORS * NUM_COLORS * NUM_TYPES,)
    factory_stacks   : np.ndarray   # (NUM_COLORS * stack_size,)
    collected        : np.ndarray   # (num_players * NUM_COLORS * 2,)
    penalty_pile     : np.ndarray   # (num_players * NUM_COLORS * NUM_TYPES,)
    scores           : np.ndarray   # (num_players,)  current scores / MAX_ECO_SCORE
    draw_pile_size   : np.ndarray   # (1,)
    draw_pile_comp   : np.ndarray   # (NUM_COLORS * NUM_TYPES,)  remaining cards by (color, type)


# ── Float dimension helper (used by ppo to size network inputs) ───────────────

def eco_float_dim(num_players: int = 2) -> int:
    stack_size = len(_STACK[num_players])
    return (
        num_players * NUM_COLORS * NUM_TYPES   # hands
        + NUM_COLORS * NUM_TYPES               # recycling_side
        + NUM_COLORS * NUM_COLORS * NUM_TYPES  # waste_side
        + NUM_COLORS * stack_size              # factory_stacks (full slot content)
        + num_players * NUM_COLORS * 2         # collected (count + value)
        + num_players * NUM_COLORS * NUM_TYPES # penalty_pile
        + num_players                          # scores
        + 1                                    # draw_pile_size
        + NUM_COLORS * NUM_TYPES              # draw_pile_comp
    )


# ── Random opponent factory ───────────────────────────────────────────────────

def _make_random_opponent(rng: np.random.Generator):
    def _opp(obs: EcoPyTreeObs, mask: np.ndarray) -> int:
        return int(rng.choice(np.where(mask)[0]))
    return _opp


# ── SinglePlayerEcoEnv ────────────────────────────────────────────────────────

class SinglePlayerEcoEnv:
    """
    Single-agent R-öko wrapper.

    The agent plays the seat assigned at reset() (player 0 by default).
    All other seats are driven by ``opponent_fn`` (defaults to random).

    Interface (gym 0.26 style):
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        mask = env.legal_actions()   # (NUM_ACTIONS,) bool
    """

    def __init__(
        self,
        num_players: int = 2,
        opponent_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
        reward_shaping_scale: float = 1.0,
        opponent_penalty: float = 0.5,
        relative_seat: bool = True,
    ):
        self.env                   = EcoEnv(num_players=num_players, seed=seed)
        self._num_players          = num_players
        self._opponent_fn_arg      = opponent_fn
        self._reward_shaping_scale = float(reward_shaping_scale)
        self._opponent_penalty     = float(opponent_penalty)
        self._relative_seat        = bool(relative_seat)
        self._seat: int            = 0
        self._shaped_reward: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> tuple:
        state = self.env.reset()
        # Randomly assign agent to any seat so training is symmetric across seats.
        # Without this the agent only ever trains as seat 0 (the first player),
        # even though it is used as the opponent from other seats during self-play.
        self._seat = int(self.env.rng.integers(0, state.num_players))
        self.opponent_fn = (
            self._opponent_fn_arg
            if self._opponent_fn_arg is not None
            else _make_random_opponent(self.env.rng)
        )
        self._shaped_reward = 0.0

        # If the agent is not the first to move, run opponents until the agent's
        # first turn (same logic as the tail of step()).
        while state.current_player != self._seat and not state.done:
            opp_obs    = self._encode_for(state.current_player)
            opp_mask   = self.env.legal_actions()
            opp_action = self.opponent_fn(opp_obs, opp_mask)
            state, rewards, terminated, _ = self.env.step(opp_action)
            self._accumulate(rewards)

        # If opponents ended the game before agent's first turn, re-reset
        if state.done:
            return self.reset()

        return self._encode(), {}

    def step(self, action: int) -> tuple:
        assert not self.env.state.done, "step() called on finished game"

        state, rewards, terminated, _ = self.env.step(action)
        self._accumulate(rewards)

        if terminated:
            return self._terminal_return(state)

        # Advance opponents until the agent's turn
        while state.current_player != self._seat:
            opp_mask   = self.env.legal_actions()
            opp_obs    = self._encode_for(state.current_player)
            opp_action = self.opponent_fn(opp_obs, opp_mask)
            state, rewards, terminated, _ = self.env.step(opp_action)
            self._accumulate(rewards)
            if terminated:
                return self._terminal_return(state)

        reward = self._shaped_reward
        self._shaped_reward = 0.0
        return self._encode(), reward, False, False, {}

    def step_gen(self, action: int):
        """Generator version of step() for batched opponent evaluation.

        Usage (by the vec env):
            gen = env.step_gen(action)
            try:
                opp_obs, opp_mask = next(gen)          # first opponent decision
                opp_obs, opp_mask = gen.send(opp_act)  # subsequent decisions
                ...
            except StopIteration as e:
                obs, reward, terminated, truncated, info = e.value

        Each yield suspends execution and hands (obs, mask) to the caller.
        The caller sends back the chosen action via gen.send(action).
        This allows the vec env to collect pending obs from ALL games before
        doing a single batched NN forward pass.
        """
        assert not self.env.state.done, "step_gen() called on finished game"

        state, rewards, terminated, _ = self.env.step(action)
        self._accumulate(rewards)

        if terminated:
            return self._terminal_return(state)

        while state.current_player != self._seat:
            opp_obs    = self._encode_for(state.current_player)
            opp_mask   = self.env.legal_actions()
            opp_action = yield (opp_obs, opp_mask)   # ← suspend; caller provides action

            state, rewards, terminated, _ = self.env.step(int(opp_action))
            self._accumulate(rewards)
            if terminated:
                return self._terminal_return(state)

        reward = self._shaped_reward
        self._shaped_reward = 0.0
        return self._encode(), reward, False, False, {}

    def legal_actions(self) -> np.ndarray:
        return self.env.legal_actions()

    @property
    def state(self) -> EcoState:
        return self.env.state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _terminal_return(self, state: EcoState) -> tuple:
        scores = self.env.compute_scores(state)
        my_score = float(scores[self._seat])
        best_score = float(scores.max())
        # Terminal: +1 win, -1 loss, 0 tie
        if my_score >= best_score:
            terminal = 1.0
        else:
            terminal = -1.0
        reward = terminal + self._shaped_reward
        self._shaped_reward = 0.0
        return self._encode(), reward, True, False, {"final_scores": scores.tolist(),
                                                       "agent_seat": self._seat}

    def _accumulate(self, rewards: np.ndarray):
        if self._reward_shaping_scale > 0.0:
            my_r = float(rewards[self._seat])
            # Penalise when opponents score — use best opponent reward as signal.
            opp_mask = [i for i in range(self._num_players) if i != self._seat]
            opp_best = max(float(rewards[i]) for i in opp_mask)
            self._shaped_reward += (my_r - self._opponent_penalty * opp_best) * self._reward_shaping_scale

    # ── Encoding ─────────────────────────────────────────────────────────────

    def _encode(self) -> EcoPyTreeObs:
        return self._encode_for(self._seat)

    def _encode_for(self, player: int) -> EcoPyTreeObs:
        s  = self.env.state
        np_ = s.num_players
        orig_stack_size = len(_STACK[self.env.num_players])

        # ── Rotate player-indexed arrays so `player` is always at index 0 ──
        # This means the network always sees "my stuff" first, removing the
        # need to learn seat-awareness from the player embedding.
        perm = [(player + i) % np_ for i in range(np_)]

        # Factory stack: full slot content, consumed slots = -1.0, active slots /= 5
        _MAX_CARD_VAL = 5.0
        factory_slots = np.full((NUM_COLORS, orig_stack_size), -1.0, dtype=np.float32)
        for c in range(NUM_COLORS):
            remaining = s.factory_stacks[c]            # list of values still in stack
            n_consumed = orig_stack_size - len(remaining)
            for i, v in enumerate(remaining):
                factory_slots[c, n_consumed + i] = v / _MAX_CARD_VAL
        factory_stacks_flat = factory_slots.flatten()

        # Hands normalised — rotated so agent's hand is first
        hands_rot = s.hands[perm]
        hands_flat = (hands_rot / SINGLES_PER_COLOR).astype(np.float32).flatten()

        # Recycling side normalised (shared, no rotation needed)
        rec_flat = (s.recycling_side / SINGLES_PER_COLOR).astype(np.float32).flatten()

        # Waste side normalised (shared, no rotation needed)
        waste_flat = (s.waste_side / SINGLES_PER_COLOR).astype(np.float32).flatten()

        # Collected factory cards per player×color — rotated
        # Two features per (player, color): scoring flag (count>1) and value sum
        coll = np.zeros((np_, NUM_COLORS, 2), dtype=np.float32)
        for idx, pp in enumerate(perm):
            for c in range(NUM_COLORS):
                cards = s.collected[pp][c]
                coll[idx, c, 0] = 1.0 if len(cards) > 1 else 0.0  # scores this color?
                coll[idx, c, 1] = sum(cards) / MAX_ECO_SCORE       # value normalised
        coll_flat = coll.flatten()

        # Penalty pile composition normalised — rotated
        pen_rot = s.penalty_pile[perm]
        pen_norm = (pen_rot / SINGLES_PER_COLOR).astype(np.float32).flatten()

        # Current scores for all players — rotated, normalised
        all_scores = self.env.compute_scores(s)
        scores_rot = (all_scores[perm] / MAX_ECO_SCORE).astype(np.float32)

        # Draw pile size normalised
        draw_norm = np.array([len(s.draw_pile) / max(_TOTAL_DECK, 1)], dtype=np.float32)

        # Draw pile composition: count of each (color, type) remaining
        draw_comp = np.zeros((NUM_COLORS, NUM_TYPES), dtype=np.float32)
        for color, typ in s.draw_pile:
            draw_comp[color, typ] += 1
        # Normalise by max possible count per type
        draw_comp[:, 0] /= max(SINGLES_PER_COLOR, 1)
        draw_comp[:, 1] /= max(DOUBLES_PER_COLOR, 1)
        draw_comp_flat = draw_comp.flatten()

        return EcoPyTreeObs(
            current_player   = np.array([0 if self._relative_seat else player + 1], dtype=np.int32),
            phase            = np.array([s.phase],    dtype=np.int32),
            hands            = hands_flat,
            recycling_side   = rec_flat,
            waste_side       = waste_flat,
            factory_stacks   = factory_stacks_flat,
            collected        = coll_flat,
            penalty_pile     = pen_norm,
            scores           = scores_rot,
            draw_pile_size   = draw_norm,
            draw_pile_comp   = draw_comp_flat,
        )
