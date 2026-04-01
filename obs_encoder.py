"""
SinglePlayerEnv: single-agent view of Hearts.

Model
-----
The environment internally runs opponents until the agent's seat needs to act.
From the agent's perspective every step() is an action decision — there are no
observation-only steps.  Rewards for tricks won by opponents while the engine
advances are accumulated and delivered to the agent on its next step (or at
terminal).

This is the correct RL formulation: the other three players are part of the
*environment*, not co-learners.  The agent trains from a single seat per
episode; the seat rotates naturally because reset() always assigns the agent
the seat that holds the 2♣ (the first player in every deal).

Opponent policy
---------------
An ``opponent_fn(obs: PyTreeObs, mask: np.ndarray) -> int`` callable is passed
at construction.  Defaults to uniform random over legal cards.

PyTreeObs fields (11 total, no phase field)
-------------------------------------------
  history_leading        : (13,)        int32   player token per completed trick
  history_pairs          : (13, 4, 2)   int32   [card_token, player_token] per slot
  current_trick_leading  : (1,)         int32   leading player of current trick
  current_trick_pairs    : (4, 2)       int32   [card_token, player_token] per slot
  current_player         : (1,)         int32   agent's seat (constant within episode)
  scores                 : (4,)         float32 normalised scores (/ MAX_SCORE)
  round                  : (1,)         float32 normalised round (/ NUM_ROUNDS)
  hand                   : (52,)        float32 agent's hand bitmap
  played                 : (52,)        float32 all cards played bitmap
  leading_suit           : (4,)         float32 one-hot suit of leading card
  hearts_broken          : (1,)         float32 0 or 1

Tokens
------
  PAD_TOKEN = 0
  Player tokens : player_id + 1  →  1-4
  Card tokens   : card_id   + 1  →  1-52
"""

import numpy as np
from typing import Optional, NamedTuple, Callable

from hearts_env import (
    HeartsEnv, HeartsState, RoundRecord,
    NUM_CARDS, NUM_PLAYERS, NUM_ROUNDS, NUM_SUITS,
    MAX_SCORE, card_suit,
)

PAD_TOKEN     = 0
PLAYER_OFFSET = 1
CARD_OFFSET   = 1

NUM_PLAYER_TOKENS = NUM_PLAYERS + 1
NUM_CARD_TOKENS   = NUM_CARDS   + 1


# ── PyTreeObs ─────────────────────────────────────────────────────────────────

class PyTreeObs(NamedTuple):
    # Embedding index fields (int32)
    history_leading       : np.ndarray  # (13,)
    history_pairs         : np.ndarray  # (13, 4, 2)
    current_trick_leading : np.ndarray  # (1,)
    current_trick_pairs   : np.ndarray  # (4, 2)
    current_player        : np.ndarray  # (1,)  agent's seat, constant per episode

    # Continuous / binary fields (float32)
    scores                : np.ndarray  # (4,)
    round                 : np.ndarray  # (1,)
    hand                  : np.ndarray  # (52,)  agent's hand
    played                : np.ndarray  # (52,)
    leading_suit          : np.ndarray  # (4,)
    hearts_broken         : np.ndarray  # (1,)


# ── Encoding helper ───────────────────────────────────────────────────────────

def _encode_trick(record: Optional[RoundRecord]):
    pairs = np.zeros((NUM_PLAYERS, 2), dtype=np.int32)
    if record is None:
        return PAD_TOKEN, pairs
    leading_token = record.leading_player + PLAYER_OFFSET
    for i in range(NUM_PLAYERS):
        c = int(record.cards[i])
        p = int(record.players[i])
        if c == -1 or p == -1:
            pairs[i] = PAD_TOKEN
        else:
            pairs[i, 0] = c + CARD_OFFSET
            pairs[i, 1] = p + PLAYER_OFFSET
    return leading_token, pairs


def _make_random_opponent(rng: np.random.Generator):
    """Return a stateless random opponent that uses the provided RNG."""
    def _opponent(obs: PyTreeObs, mask: np.ndarray) -> int:
        return int(rng.choice(np.where(mask)[0]))
    return _opponent


# ── SinglePlayerEnv ───────────────────────────────────────────────────────────

class SinglePlayerEnv:
    """
    Single-agent Hearts environment.

    The agent always plays the seat that holds the 2♣ at the start of each
    deal (i.e. the first player).  The other three seats are controlled by
    ``opponent_fn``.

    Interface (gym 0.26)
    --------------------
    obs, info            = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    mask                 = env.legal_actions()   # (52,) bool

    Reward
    ------
    Accumulated trick rewards for the agent's seat, delivered when the agent
    next needs to act (or at game end).  ACT steps return the reward that
    accrued since the last time the agent acted.
    """

    def __init__(
        self,
        opponent_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
        reward_shaping_scale: float = 0.0,
    ):
        self.env         = HeartsEnv(seed=seed)
        # If no opponent supplied, use a random policy driven by the env's own RNG
        # so that seeding is fully reproducible without touching global numpy state.
        self._opponent_fn_arg    = opponent_fn   # None means "use env rng"
        self._reward_shaping_scale = float(reward_shaping_scale)
        self._seat: int  = 0
        self._shaped_reward: float = 0.0  # accumulates per-trick shaping between agent steps

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> tuple:
        state = self.env.reset()
        self._seat = int(state.current_player)   # holder of 2♣
        # Build opponent fn once per episode so env rng is current
        if self._opponent_fn_arg is not None:
            self.opponent_fn = self._opponent_fn_arg
        else:
            self.opponent_fn = _make_random_opponent(self.env.rng)
        self._shaped_reward = 0.0
        return self._encode(), {}

    def step(self, action: int) -> tuple:
        """Play agent's action, advance opponents, return next agent step.

        Reward policy
        -------------
        Terminal reward: _terminal_rewards[seat] — normalised final score,
        handles shoot-the-moon.  Always delivered; never scaled.

        Optional shaping (reward_shaping_scale > 0):
        Each time a trick resolves (agent or opponent), the points taken
        by the agent's seat contribute -(trick_points / MAX_SCORE) * scale
        to an accumulator that is added to the next reward returned.
        This provides per-trick signal without replacing the terminal reward.
        """
        assert not self.env.state.done, "step() called on finished game"

        # Agent plays
        state, trick_rewards, terminated, _ = self.env.step(action)
        self._accumulate_shaping(trick_rewards)

        if terminated:
            reward = float(self.env._terminal_rewards(state)[self._seat])
            reward += self._shaped_reward
            self._shaped_reward = 0.0
            return self._encode(), reward, True, False, {"final_scores": state.scores.copy()}

        # Advance opponents until it's the agent's turn again
        while state.current_player != self._seat:
            opp_obs    = self._encode_for(state.current_player)
            opp_mask   = self._legal_mask()
            opp_action = self.opponent_fn(opp_obs, opp_mask)
            state, trick_rewards, terminated, _ = self.env.step(opp_action)
            self._accumulate_shaping(trick_rewards)
            if terminated:
                reward = float(self.env._terminal_rewards(state)[self._seat])
                reward += self._shaped_reward
                self._shaped_reward = 0.0
                return self._encode(), reward, True, False, {"final_scores": state.scores.copy()}

        # Deliver accumulated shaping for this agent step
        reward = self._shaped_reward
        self._shaped_reward = 0.0
        return self._encode(), reward, False, False, {}

    def _accumulate_shaping(self, trick_rewards: np.ndarray) -> None:
        """Add scaled per-trick reward for agent's seat to accumulator."""
        if self._reward_shaping_scale > 0.0:
            self._shaped_reward += float(trick_rewards[self._seat]) * self._reward_shaping_scale

    def legal_actions(self) -> np.ndarray:
        """Return (52,) bool mask of legal cards for the agent."""
        return self.env.legal_actions()

    # ── Encoding ─────────────────────────────────────────────────────────────

    def _encode(self) -> PyTreeObs:
        return self._encode_for(self._seat)

    def _encode_for(self, player: int) -> PyTreeObs:
        s = self.env.state

        history_leading = np.zeros(NUM_ROUNDS, dtype=np.int32)
        history_pairs   = np.zeros((NUM_ROUNDS, NUM_PLAYERS, 2), dtype=np.int32)
        for i, record in enumerate(s.history):
            tok, pairs = _encode_trick(record)
            history_leading[i] = tok
            history_pairs[i]   = pairs

        if s.current_trick_count > 0:
            curr_rec = RoundRecord(
                leading_player=s.leading_player,
                cards=s.current_trick_cards,
                players=s.current_trick_players,
            )
            curr_tok, curr_pairs = _encode_trick(curr_rec)
        else:
            curr_tok  = PAD_TOKEN
            curr_pairs = np.zeros((NUM_PLAYERS, 2), dtype=np.int32)

        leading_suit = np.zeros(NUM_SUITS, dtype=np.float32)
        ls = s.leading_suit()
        if ls >= 0:
            leading_suit[ls] = 1.0

        return PyTreeObs(
            history_leading       = history_leading,
            history_pairs         = history_pairs,
            current_trick_leading = np.array([curr_tok], dtype=np.int32),
            current_trick_pairs   = curr_pairs,
            current_player        = np.array([self._seat], dtype=np.int32),
            scores                = (s.scores / MAX_SCORE).astype(np.float32),
            round                 = np.array([s.round_num / NUM_ROUNDS], dtype=np.float32),
            hand                  = s.hands[player].astype(np.float32),
            played                = s.played_cards.astype(np.float32),
            leading_suit          = leading_suit,
            hearts_broken         = np.array([float(s.hearts_broken)], dtype=np.float32),
        )

    def _legal_mask(self) -> np.ndarray:
        """52-card mask for whoever is currently acting (used for opponents)."""
        return self.env.legal_actions()

    # Keep old name for compatibility
    @property
    def state(self):
        return self.env.state


# Backwards-compat alias
HeartsEnvWrapper = SinglePlayerEnv