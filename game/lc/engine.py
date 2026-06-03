"""
Lost Cities — BaseGameEngine implementation (Numba jitclass-accelerated).

2-player card game. 60 cards: 5 colors × 12 cards (3 investments + numbered 2–10).
Each turn: play a card (expedition or discard) then draw a card (deck or discard pile).
Scoring: for each expedition, (sum_values - penalty) × (1 + num_investments), +20 bonus if ≥8 cards.

Action encoding (600 total):
  flat = card_id * 12 + action_type * 6 + draw_source
  card_id ∈ [0, 50)  — 5 colors × 10 unique values
  action_type ∈ {0=expedition, 1=discard}
  draw_source ∈ [0, 6) — 0=deck, 1–5=discard piles by color

Architecture:
  LCState (@jitclass)  — pure game state + compiled methods (step, encode, legal, reset)
  LCEngine (Python)    — thin wrapper for BaseGameEngine interface + RNG
  batch_* (@njit)      — loop over typed.List[LCState] in compiled code
"""

from typing import NamedTuple
import numpy as np
import numba as nb
from numba.typed import List as NbList
from abstract.game import BaseGameEngine

# ── Constants ───────────────────────────────────────────────────────────────

NUM_COLORS = 5
NUM_UNIQUE_VALUES = 10  # 0(investment), 2,3,4,5,6,7,8,9,10
NUM_CARD_IDS = NUM_COLORS * NUM_UNIQUE_VALUES  # 50
NUM_CARDS_PER_COLOR = 12  # 3 investments + 9 numbered
TOTAL_CARDS = NUM_COLORS * NUM_CARDS_PER_COLOR  # 60
HAND_SIZE = 8
NUM_PLAYERS = 2
DEFAULT_NEW_COLOR_PENALTY = 20
EIGHT_CARD_BONUS = 20

# Action space
NUM_ACTION_TYPES = 2   # 0=expedition, 1=discard
NUM_DRAW_SOURCES = 6   # 0=deck, 1-5=discard color 0-4
NUM_ACTIONS = NUM_CARD_IDS * NUM_ACTION_TYPES * NUM_DRAW_SOURCES  # 600

COLOR_NAMES = ["Blue", "Yellow", "White", "Green", "Red"]

# Phases (for decomposed action mode — 2-phase)
PHASE_PLAY = 0   # choose card + expedition/discard
PHASE_DRAW = 1   # choose draw source

# Decomposed action space (106 total, phase-dependent masking):
NUM_PLAY_ACTIONS = NUM_CARD_IDS * NUM_ACTION_TYPES  # 100
NUM_DRAW_ACTIONS = NUM_DRAW_SOURCES                  # 6
NUM_DECOMPOSED_ACTIONS = NUM_PLAY_ACTIONS + NUM_DRAW_ACTIONS  # 106

# Phases (for 3-phase decomposed action mode)
PHASE3_SELECT_CARD = 0
PHASE3_ACTION_TYPE = 1
PHASE3_DRAW = 2

# 3-phase decomposed action space (58 total):
NUM_CARD_SELECT_ACTIONS = NUM_CARD_IDS               # 50
NUM_TYPE_ACTIONS = NUM_ACTION_TYPES                   # 2
NUM_3PHASE_DRAW_ACTIONS = NUM_DRAW_SOURCES            # 6
NUM_3PHASE_ACTIONS = NUM_CARD_SELECT_ACTIONS + NUM_TYPE_ACTIONS + NUM_3PHASE_DRAW_ACTIONS  # 58

# Card values
def _card_value(value_idx: int) -> int:
    return 0 if value_idx == 0 else value_idx + 1

CARD_VALUES = [_card_value(i) for i in range(NUM_UNIQUE_VALUES)]
_CARD_VALUES_ARR = np.array(CARD_VALUES, dtype=np.int32)
CARD_COPIES = [3] + [1] * 9

# Pre-built deck template
_DECK_TEMPLATE = np.empty(TOTAL_CARDS, dtype=np.int8)
_idx = 0
for _c in range(NUM_COLORS):
    for _v in range(NUM_UNIQUE_VALUES):
        for _ in range(CARD_COPIES[_v]):
            _DECK_TEMPLATE[_idx] = _c * NUM_UNIQUE_VALUES + _v
            _idx += 1


# ── Action encoding ────────────────────────────────────────────────────────

def encode_action(card_id: int, action_type: int, draw_source: int) -> int:
    return card_id * 12 + action_type * 6 + draw_source

def decode_action(flat: int) -> tuple[int, int, int]:
    draw_source = flat % 6
    action_type = (flat // 6) % 2
    card_id = flat // 12
    return card_id, action_type, draw_source

def card_id_to_color_value(card_id: int) -> tuple[int, int]:
    return card_id // NUM_UNIQUE_VALUES, card_id % NUM_UNIQUE_VALUES

def color_value_to_card_id(color_idx: int, value_idx: int) -> int:
    return color_idx * NUM_UNIQUE_VALUES + value_idx


# ── Decomposed action encoding ────────────────────────────────────────────

def encode_play_action(card_id: int, action_type: int) -> int:
    return card_id * 2 + action_type

def decode_play_action(action: int) -> tuple[int, int]:
    return action // 2, action % 2

def encode_draw_action(draw_source: int) -> int:
    return NUM_PLAY_ACTIONS + draw_source

def decode_draw_action(action: int) -> int:
    return action - NUM_PLAY_ACTIONS

def encode_card_select_action(card_id: int) -> int:
    return card_id

def decode_card_select_action(action: int) -> int:
    return action

def encode_type_action(action_type: int) -> int:
    return NUM_CARD_SELECT_ACTIONS + action_type

def decode_type_action(action: int) -> int:
    return action - NUM_CARD_SELECT_ACTIONS

def encode_3phase_draw_action(draw_source: int) -> int:
    return NUM_CARD_SELECT_ACTIONS + NUM_TYPE_ACTIONS + draw_source

def decode_3phase_draw_action(action: int) -> int:
    return action - NUM_CARD_SELECT_ACTIONS - NUM_TYPE_ACTIONS


# ── Observation ─────────────────────────────────────────────────────────────

class LCObs(NamedTuple):
    """Lost Cities observation: float32 arrays + phase (int32)."""
    hand: np.ndarray              # (50,) card counts in hand / 3
    own_expeditions: np.ndarray   # (50,) cards in own expeditions / 3
    opp_expeditions: np.ndarray   # (50,) cards in opponent expeditions / 3
    discard_top1: np.ndarray      # (50,) top card of each discard pile
    discard_top2: np.ndarray      # (50,) 2nd card of each discard pile
    discard_top3: np.ndarray      # (50,) 3rd card of each discard pile
    deck_size: np.ndarray         # (1,) deck_size / 44 (max after dealing)
    scores: np.ndarray            # (2,) current scores / 100, rotated [own, opp]
    phase: np.ndarray             # (1,) int32
    played_card: np.ndarray       # (NUM_CARD_IDS,) one-hot of card selected/played
    played_type: np.ndarray       # (1,) 0=expedition, 1=discard (-1 default)

def float_dim(decompose_actions: bool = False, three_phase: bool = False) -> int:
    base = 50 * 6 + 1 + 2  # 303
    if decompose_actions or three_phase:
        base += NUM_CARD_IDS + 1
    return base


# ── Numba RNG seed ──────────────────────────────────────────────────────────
# Numba uses a process-global (thread-local) PRNG for np.random calls inside
# @njit / jitclass.  All 128 envs in a process share it sequentially — fine.
# But each process (multiprocessing) MUST call seed_numba_rng() with a unique
# seed at startup to avoid identical shuffle sequences across workers.

@nb.njit
def seed_numba_rng(seed):
    """Seed Numba's internal PRNG. Call once per process with a unique seed."""
    np.random.seed(seed)


# ── LCState jitclass ────────────────────────────────────────────────────────

_lc_state_spec = [
    # Mutable game state
    ('hands',             nb.int8[:, :]),       # (2, 50)
    ('exp_vals',          nb.int8[:, :, :]),    # (2, 5, 12)
    ('exp_len',           nb.int8[:, :]),       # (2, 5)
    ('discard',           nb.int8[:, :]),       # (5, 12)
    ('discard_len',       nb.int8[:]),          # (5,)
    ('deck',              nb.int8[:]),          # (60,)
    ('deck_top',          nb.int32),
    ('current_player',    nb.int8),
    ('is_done',           nb.boolean),
    ('scores',            nb.float32[:]),       # (2,)
    ('prev_scores',       nb.float32[:]),       # (2,)
    ('discard_draws',     nb.int32[:]),         # (2,)
    # Decomposed phase state
    ('phase',             nb.int8),
    ('played_color',      nb.int8),
    ('played_val_idx',    nb.int8),
    ('played_card_id',    nb.int8),
    ('played_action_type', nb.int8),
    # Config (immutable after init)
    ('penalty',           nb.int32),
    ('max_lanes',         nb.int8),
    ('max_dd',            nb.int32),           # max_discard_draws
    ('dense_reward',      nb.boolean),
    ('score_diff_reward', nb.boolean),
    ('zero_one_reward',   nb.boolean),
    ('raw_score_reward',  nb.boolean),
    ('dense_opp_delta',   nb.boolean),
    ('dense_opp_penalty', nb.boolean),
    # Constants
    ('deck_template',     nb.int8[:]),         # (60,)
    ('card_values',       nb.int32[:]),        # (10,)
]


@nb.experimental.jitclass(_lc_state_spec)
class LCState:

    def __init__(self, penalty, max_lanes, max_dd,
                 dense_reward, score_diff_reward, zero_one_reward,
                 raw_score_reward, dense_opp_delta, dense_opp_penalty,
                 deck_template, card_values):
        self.penalty = penalty
        self.max_lanes = max_lanes
        self.max_dd = max_dd
        self.dense_reward = dense_reward
        self.score_diff_reward = score_diff_reward
        self.zero_one_reward = zero_one_reward
        self.raw_score_reward = raw_score_reward
        self.dense_opp_delta = dense_opp_delta
        self.dense_opp_penalty = dense_opp_penalty
        self.deck_template = deck_template
        self.card_values = card_values
        # Allocate state arrays
        self.hands = np.zeros((2, 50), dtype=np.int8)
        self.exp_vals = np.zeros((2, 5, 12), dtype=np.int8)
        self.exp_len = np.zeros((2, 5), dtype=np.int8)
        self.discard = np.zeros((5, 12), dtype=np.int8)
        self.discard_len = np.zeros(5, dtype=np.int8)
        self.deck = np.empty(60, dtype=np.int8)
        self.deck_top = nb.int32(0)
        self.current_player = nb.int8(0)
        self.is_done = False
        self.scores = np.zeros(2, dtype=np.float32)
        self.prev_scores = np.zeros(2, dtype=np.float32)
        self.discard_draws = np.zeros(2, dtype=np.int32)
        self.phase = nb.int8(0)
        self.played_color = nb.int8(-1)
        self.played_val_idx = nb.int8(-1)
        self.played_card_id = nb.int8(-1)
        self.played_action_type = nb.int8(-1)

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self):
        self.deck[:] = self.deck_template
        np.random.shuffle(self.deck)
        self.deck_top = nb.int32(60)
        self.hands[:] = 0
        self.exp_vals[:] = 0
        self.exp_len[:] = 0
        self.discard[:] = 0
        self.discard_len[:] = 0
        self.current_player = nb.int8(0)
        self.is_done = False
        self.scores[:] = 0.0
        self.prev_scores[:] = 0.0
        self.discard_draws[:] = 0
        self.phase = nb.int8(0)
        self.played_color = nb.int8(-1)
        self.played_val_idx = nb.int8(-1)
        self.played_card_id = nb.int8(-1)
        self.played_action_type = nb.int8(-1)
        # Deal hands
        for _ in range(8):
            for p in range(2):
                self.deck_top -= 1
                cid = self.deck[self.deck_top]
                self.hands[p, cid] += 1

    # ── Scoring ────────────────────────────────────────────────────────

    def calc_scores(self):
        """Calculate scores for both players, write into self.scores."""
        for p in range(2):
            total = nb.int32(0)
            for c in range(5):
                n = self.exp_len[p, c]
                if n == 0:
                    continue
                base_value = nb.int32(0)
                num_inv = nb.int32(0)
                for j in range(n):
                    v = self.exp_vals[p, c, j]
                    if v == 0:
                        num_inv += 1
                    else:
                        base_value += self.card_values[v]
                multiplier = num_inv + 1
                exp_score = (base_value - self.penalty) * multiplier
                if n >= 8:
                    exp_score += 20
                total += exp_score
            self.scores[p] = nb.float32(total)

    # ── Reward assignment ──────────────────────────────────────────────

    def _assign_terminal(self, r):
        if self.raw_score_reward:
            r[0] = self.scores[0] / 30.0
            r[1] = self.scores[1] / 30.0
        elif self.score_diff_reward:
            r[0] = (self.scores[0] - self.scores[1]) / 30.0
            r[1] = (self.scores[1] - self.scores[0]) / 30.0
        elif self.zero_one_reward:
            best = max(self.scores[0], self.scores[1])
            r[0] = 1.0 if self.scores[0] >= best else 0.0
            r[1] = 1.0 if self.scores[1] >= best else 0.0
        else:
            best = max(self.scores[0], self.scores[1])
            worst = min(self.scores[0], self.scores[1])
            for i in range(2):
                if self.scores[i] >= best:
                    r[i] = 1.0
                elif self.scores[i] <= worst:
                    r[i] = -1.0
                else:
                    r[i] = 0.0

    def _assign_dense_terminal(self, r):
        if self.dense_opp_penalty:
            for i in range(2):
                opp = 1 - i
                delta = (self.scores[i] - self.prev_scores[i]) / 30.0
                r[i] = delta - self.scores[opp] / 30.0
        elif self.dense_opp_delta or self.score_diff_reward:
            for i in range(2):
                opp = 1 - i
                r[i] = ((self.scores[i] - self.prev_scores[i]) -
                        (self.scores[opp] - self.prev_scores[opp])) / 30.0
        elif self.zero_one_reward:
            best = max(self.scores[0], self.scores[1])
            for i in range(2):
                delta = (self.scores[i] - self.prev_scores[i]) / 30.0
                bonus = 1.0 if self.scores[i] >= best else 0.0
                r[i] = delta + bonus
        elif self.raw_score_reward:
            for i in range(2):
                r[i] = (self.scores[i] - self.prev_scores[i]) / 30.0
        else:
            best = max(self.scores[0], self.scores[1])
            worst = min(self.scores[0], self.scores[1])
            for i in range(2):
                delta = (self.scores[i] - self.prev_scores[i]) / 30.0
                if self.scores[i] >= best:
                    bonus = 1.0
                elif self.scores[i] <= worst:
                    bonus = -1.0
                else:
                    bonus = 0.0
                r[i] = delta + bonus

    # ── Step (flat 600-action) ─────────────────────────────────────────

    def step_flat(self, action, r):
        """Step with flat action, write 2-player rewards into r[0:2]."""
        r[0] = 0.0
        r[1] = 0.0
        card_id = action // 12
        action_type = (action // 6) % 2
        draw_source = action % 6
        p = self.current_player
        color = card_id // 10
        val_idx = card_id % 10

        # Play
        self.hands[p, card_id] -= 1
        if action_type == 0:
            n = self.exp_len[p, color]
            self.exp_vals[p, color, n] = nb.int8(val_idx)
            self.exp_len[p, color] = n + 1
        else:
            n = self.discard_len[color]
            self.discard[color, n] = nb.int8(card_id)
            self.discard_len[color] = n + 1

        # Draw
        if draw_source == 0:
            self.deck_top -= 1
            drawn = self.deck[self.deck_top]
            self.hands[p, drawn] += 1
        else:
            pile_c = draw_source - 1
            self.discard_len[pile_c] -= 1
            drawn = self.discard[pile_c, self.discard_len[pile_c]]
            self.hands[p, drawn] += 1
            self.discard_draws[p] += 1

        # Terminal
        if self.deck_top == 0:
            self.is_done = True
            self.calc_scores()
            if self.dense_reward:
                self._assign_dense_terminal(r)
            else:
                self._assign_terminal(r)
        else:
            if self.dense_reward:
                prev0 = self.prev_scores[0]
                prev1 = self.prev_scores[1]
                self.calc_scores()
                cur0 = self.scores[0]
                cur1 = self.scores[1]
                if self.dense_opp_delta or self.score_diff_reward:
                    for i in range(2):
                        opp = 1 - i
                        oi = self.scores[i] - self.prev_scores[i]
                        oo = self.scores[opp] - self.prev_scores[opp]
                        r[i] = (oi - oo) / 30.0
                else:
                    r[p] = (self.scores[p] - self.prev_scores[p]) / 30.0
                self.prev_scores[0] = cur0
                self.prev_scores[1] = cur1
            self.current_player = nb.int8(1 - p)

    # ── Step (decomposed 2-phase) ──────────────────────────────────────

    def step_decomposed(self, action, r):
        if self.phase == 0:  # PHASE_PLAY
            self._step_play(action, r)
        else:
            self._step_draw(action, r)

    def _step_play(self, action, r):
        r[0] = 0.0
        r[1] = 0.0
        card_id = action // 2
        action_type = action % 2
        p = self.current_player
        color = card_id // 10
        val_idx = card_id % 10
        self.hands[p, card_id] -= 1
        if action_type == 0:
            n = self.exp_len[p, color]
            self.exp_vals[p, color, n] = nb.int8(val_idx)
            self.exp_len[p, color] = n + 1
        else:
            n = self.discard_len[color]
            self.discard[color, n] = nb.int8(card_id)
            self.discard_len[color] = n + 1
        self.played_color = nb.int8(color)
        self.played_val_idx = nb.int8(val_idx)
        self.played_action_type = nb.int8(action_type)
        self.phase = nb.int8(1)  # PHASE_DRAW

    def _step_draw(self, action, r):
        r[0] = 0.0
        r[1] = 0.0
        draw_source = action - 100  # NUM_PLAY_ACTIONS
        p = self.current_player
        if draw_source == 0:
            self.deck_top -= 1
            drawn = self.deck[self.deck_top]
            self.hands[p, drawn] += 1
        else:
            pile_c = draw_source - 1
            self.discard_len[pile_c] -= 1
            drawn = self.discard[pile_c, self.discard_len[pile_c]]
            self.hands[p, drawn] += 1
            self.discard_draws[p] += 1
        if self.deck_top == 0:
            self.is_done = True
            self.calc_scores()
            self._assign_terminal(r)
        else:
            self.current_player = nb.int8(1 - p)
        self.phase = nb.int8(0)  # PHASE_PLAY

    # ── Step (3-phase) ─────────────────────────────────────────────────

    def step_3phase(self, action, r):
        if self.phase == 0:  # SELECT_CARD
            r[0] = 0.0; r[1] = 0.0
            card_id = action
            color = card_id // 10
            val_idx = card_id % 10
            self.played_card_id = nb.int8(card_id)
            self.played_color = nb.int8(color)
            self.played_val_idx = nb.int8(val_idx)
            self.phase = nb.int8(1)
        elif self.phase == 1:  # ACTION_TYPE
            r[0] = 0.0; r[1] = 0.0
            action_type = action - 50  # NUM_CARD_SELECT_ACTIONS
            p = self.current_player
            card_id = self.played_card_id
            color = self.played_color
            val_idx = self.played_val_idx
            self.hands[p, card_id] -= 1
            if action_type == 0:
                n = self.exp_len[p, color]
                self.exp_vals[p, color, n] = nb.int8(val_idx)
                self.exp_len[p, color] = n + 1
            else:
                n = self.discard_len[color]
                self.discard[color, n] = nb.int8(card_id)
                self.discard_len[color] = n + 1
            self.played_action_type = nb.int8(action_type)
            self.phase = nb.int8(2)
        else:  # DRAW
            r[0] = 0.0; r[1] = 0.0
            draw_source = action - 52  # NUM_CARD_SELECT + NUM_TYPE
            p = self.current_player
            if draw_source == 0:
                self.deck_top -= 1
                drawn = self.deck[self.deck_top]
                self.hands[p, drawn] += 1
            else:
                pile_c = draw_source - 1
                self.discard_len[pile_c] -= 1
                drawn = self.discard[pile_c, self.discard_len[pile_c]]
                self.hands[p, drawn] += 1
                self.discard_draws[p] += 1
            if self.deck_top == 0:
                self.is_done = True
                self.calc_scores()
                self._assign_terminal(r)
            else:
                self.current_player = nb.int8(1 - p)
            self.phase = nb.int8(0)
            self.played_card_id = nb.int8(-1)
            self.played_color = nb.int8(-1)
            self.played_val_idx = nb.int8(-1)
            self.played_action_type = nb.int8(-1)

    # ── Legal actions ──────────────────────────────────────────────────

    def _dd_blocked(self):
        if self.max_dd < 0:
            return True
        if self.max_dd == 0:
            return False
        return (self.discard_draws[0] + self.discard_draws[1]) >= self.max_dd

    def _is_valid_exp(self, player, card_id):
        color = card_id // 10
        val_idx = card_id % 10
        n = self.exp_len[player, color]
        if n == 0:
            return True
        last = self.exp_vals[player, color, n - 1]
        if val_idx == 0:
            num_inv = nb.int32(0)
            for j in range(n):
                if self.exp_vals[player, color, j] == 0:
                    num_inv += 1
            return num_inv < 3 and last == 0
        return val_idx > last

    def fill_legal_flat(self, mask):
        mask[:] = False
        p = self.current_player
        block = self._dd_blocked()
        open_lanes = nb.int32(0)
        for c in range(5):
            if self.exp_len[p, c] > 0:
                open_lanes += 1
        for card_id in range(50):
            if self.hands[p, card_id] <= 0:
                continue
            color = card_id // 10
            # Expedition
            can_exp = False
            n = self.exp_len[p, color]
            if (n > 0 or open_lanes < self.max_lanes):
                if self._is_valid_exp(p, card_id):
                    can_exp = True
            if can_exp:
                base = card_id * 12
                if self.deck_top > 0:
                    mask[base] = True
                if not block:
                    for ds in range(1, 6):
                        if self.discard_len[ds - 1] > 0:
                            mask[base + ds] = True
            # Discard
            base = card_id * 12 + 6
            if self.deck_top > 0:
                mask[base] = True
            if not block:
                for ds in range(1, 6):
                    if ds - 1 == color:
                        continue
                    if self.discard_len[ds - 1] > 0:
                        mask[base + ds] = True

    def fill_legal_decomposed(self, mask):
        mask[:] = False
        p = self.current_player
        if self.phase == 0:  # PLAY
            open_lanes = nb.int32(0)
            for c in range(5):
                if self.exp_len[p, c] > 0:
                    open_lanes += 1
            for card_id in range(50):
                if self.hands[p, card_id] <= 0:
                    continue
                n = self.exp_len[p, card_id // 10]
                if (n > 0 or open_lanes < self.max_lanes) and self._is_valid_exp(p, card_id):
                    mask[card_id * 2] = True
                mask[card_id * 2 + 1] = True
        else:  # DRAW
            if self.deck_top > 0:
                mask[100] = True
            if not self._dd_blocked():
                for c in range(5):
                    if self.played_action_type == 1 and c == self.played_color:
                        continue
                    if self.discard_len[c] > 0:
                        mask[101 + c] = True

    def fill_legal_3phase(self, mask):
        mask[:] = False
        p = self.current_player
        if self.phase == 0:  # SELECT
            for card_id in range(50):
                if self.hands[p, card_id] > 0:
                    mask[card_id] = True
        elif self.phase == 1:  # TYPE
            open_lanes = nb.int32(0)
            for c in range(5):
                if self.exp_len[p, c] > 0:
                    open_lanes += 1
            n = self.exp_len[p, self.played_color]
            if (n > 0 or open_lanes < self.max_lanes) and self._is_valid_exp(p, self.played_card_id):
                mask[50] = True
            mask[51] = True
        else:  # DRAW
            if self.deck_top > 0:
                mask[52] = True
            if not self._dd_blocked():
                for c in range(5):
                    if self.played_action_type == 1 and c == self.played_color:
                        continue
                    if self.discard_len[c] > 0:
                        mask[53 + c] = True

    # ── Encode observation ─────────────────────────────────────────────

    def encode_obs(self, player_id,
                   hand_out, own_exp_out, opp_exp_out,
                   dt1_out, dt2_out, dt3_out,
                   deck_size_out, scores_out, phase_out,
                   played_card_out, played_type_out):
        opp = 1 - player_id
        for i in range(50):
            hand_out[i] = self.hands[player_id, i] / 3.0
        own_exp_out[:] = 0.0
        opp_exp_out[:] = 0.0
        for c in range(5):
            for j in range(self.exp_len[player_id, c]):
                cid = c * 10 + self.exp_vals[player_id, c, j]
                own_exp_out[cid] += 1.0 / 3.0
            for j in range(self.exp_len[opp, c]):
                cid = c * 10 + self.exp_vals[opp, c, j]
                opp_exp_out[cid] += 1.0 / 3.0
        dt1_out[:] = 0.0
        dt2_out[:] = 0.0
        dt3_out[:] = 0.0
        for c in range(5):
            n = self.discard_len[c]
            if n >= 1:
                dt1_out[self.discard[c, n - 1]] = 1.0
            if n >= 2:
                dt2_out[self.discard[c, n - 2]] = 1.0
            if n >= 3:
                dt3_out[self.discard[c, n - 3]] = 1.0
        deck_size_out[0] = self.deck_top / 44.0
        # Scores for obs (not game scores): own/100, opp/100
        self.calc_scores()
        scores_out[0] = self.scores[player_id] / 100.0
        scores_out[1] = self.scores[opp] / 100.0
        phase_out[0] = self.phase
        played_card_out[:] = 0.0
        played_type_out[0] = -1.0
        if self.played_color >= 0:
            played_card_out[self.played_color * 10 + self.played_val_idx] = 1.0
            if self.played_action_type >= 0:
                played_type_out[0] = nb.float32(self.played_action_type)


# ── Batch @njit functions ──────────────────────────────────────────────────

@nb.njit
def batch_step_agents(states, actions, seats, acc_rewards, rbuf):
    """Step all states with agent actions, accumulate seat rewards."""
    for i in range(len(states)):
        s = states[i]
        if s.is_done:
            continue
        s.step_flat(actions[i], rbuf)
        acc_rewards[i] += rbuf[seats[i]]


@nb.njit
def batch_find_and_encode_opponents(states, seats, pending_out,
                                    hand, own_exp, opp_exp,
                                    dt1, dt2, dt3,
                                    deck_size, scores, phase,
                                    played_card, played_type,
                                    mask):
    """Find envs needing opponent turn, encode obs + legal masks contiguously."""
    n = nb.int32(0)
    for i in range(len(states)):
        s = states[i]
        if s.is_done or s.current_player == seats[i]:
            continue
        pending_out[n] = nb.int32(i)
        p = s.current_player
        s.encode_obs(p, hand[n], own_exp[n], opp_exp[n],
                     dt1[n], dt2[n], dt3[n],
                     deck_size[n], scores[n], phase[n],
                     played_card[n], played_type[n])
        s.fill_legal_flat(mask[n])
        n += 1
    return n


@nb.njit
def batch_step_pending(states, pending, n_pending, actions,
                       seats, acc_rewards, rbuf):
    """Step pending states with opponent actions."""
    for j in range(n_pending):
        i = pending[j]
        s = states[i]
        s.step_flat(actions[j], rbuf)
        acc_rewards[i] += rbuf[seats[i]]


@nb.njit
def batch_collect_results(states, acc_rewards, rewards_out, terminated_out, done_indices):
    """Copy rewards and check done flags. Returns number of done envs."""
    n_done = nb.int32(0)
    for i in range(len(states)):
        rewards_out[i] = nb.float32(acc_rewards[i])
        if states[i].is_done:
            terminated_out[i] = True
            done_indices[n_done] = nb.int32(i)
            n_done += 1
        else:
            terminated_out[i] = False
    return n_done


@nb.njit
def batch_encode_agents(states, seats,
                        hand, own_exp, opp_exp,
                        dt1, dt2, dt3,
                        deck_size, scores, phase,
                        played_card, played_type,
                        mask):
    """Encode agent obs + legal masks for all states."""
    for i in range(len(states)):
        s = states[i]
        p = seats[i]
        s.encode_obs(p, hand[i], own_exp[i], opp_exp[i],
                     dt1[i], dt2[i], dt3[i],
                     deck_size[i], scores[i], phase[i],
                     played_card[i], played_type[i])
        s.fill_legal_flat(mask[i])


# ── LCEngine — Python wrapper ──────────────────────────────────────────────

class LCEngine(BaseGameEngine[LCObs]):
    """Lost Cities engine. Wraps LCState jitclass for BaseGameEngine interface."""

    SCORE_DIFF_NORM = 30.0

    def __init__(self, rng: np.random.Generator,
                 new_color_penalty: int = DEFAULT_NEW_COLOR_PENALTY,
                 score_diff_reward: bool = False,
                 zero_one_reward: bool = False,
                 max_lanes: int = NUM_COLORS,
                 decompose_actions: bool = False,
                 three_phase: bool = False,
                 max_discard_draws: int = 0,
                 raw_score_reward: bool = False,
                 dense_reward: bool = False,
                 dense_opponent_delta: bool = False,
                 dense_opp_penalty: bool = False):
        super().__init__(rng)
        self._decompose_actions = decompose_actions
        self._three_phase = three_phase
        self._state = LCState(
            penalty=new_color_penalty,
            max_lanes=max_lanes,
            max_dd=max_discard_draws,
            dense_reward=dense_reward,
            score_diff_reward=score_diff_reward,
            zero_one_reward=zero_one_reward,
            raw_score_reward=raw_score_reward,
            dense_opp_delta=dense_opponent_delta,
            dense_opp_penalty=dense_opp_penalty,
            deck_template=_DECK_TEMPLATE.copy(),
            card_values=_CARD_VALUES_ARR.copy(),
        )
        self._reward_buf = np.zeros(2, dtype=np.float64)
        if three_phase:
            self._mask_buf = np.zeros(NUM_3PHASE_ACTIONS, dtype=np.bool_)
        elif decompose_actions:
            self._mask_buf = np.zeros(NUM_DECOMPOSED_ACTIONS, dtype=np.bool_)
        else:
            self._mask_buf = np.zeros(NUM_ACTIONS, dtype=np.bool_)

    @property
    def _scores(self) -> np.ndarray:
        return self._state.scores

    @property
    def _hands(self):
        return self._state.hands

    @property
    def _exp_vals(self):
        return self._state.exp_vals

    @property
    def _exp_len(self):
        return self._state.exp_len

    @property
    def _discard(self):
        return self._state.discard

    @property
    def _discard_len(self):
        return self._state.discard_len

    @property
    def _discard_draws(self):
        return self._state.discard_draws

    def _reset(self) -> None:
        self._state.reset()

    @property
    def current_player(self) -> int:
        return int(self._state.current_player)

    @property
    def done(self) -> bool:
        return self._state.is_done

    @property
    def num_players(self) -> int:
        return NUM_PLAYERS

    @property
    def num_actions(self) -> int:
        if self._three_phase:
            return NUM_3PHASE_ACTIONS
        if self._decompose_actions:
            return NUM_DECOMPOSED_ACTIONS
        return NUM_ACTIONS

    def step(self, action: int) -> tuple[float, ...]:
        r = self._reward_buf
        s = self._state
        if self._three_phase:
            s.step_3phase(action, r)
        elif self._decompose_actions:
            s.step_decomposed(action, r)
        else:
            s.step_flat(action, r)
        return (float(r[0]), float(r[1]))

    def legal_actions(self) -> np.ndarray:
        s = self._state
        m = self._mask_buf
        if self._three_phase:
            s.fill_legal_3phase(m)
        elif self._decompose_actions:
            s.fill_legal_decomposed(m)
        else:
            s.fill_legal_flat(m)
        return m

    def encode(self, player_id: int) -> LCObs:
        s = self._state
        hand = np.zeros(50, dtype=np.float32)
        own_exp = np.zeros(50, dtype=np.float32)
        opp_exp = np.zeros(50, dtype=np.float32)
        dt1 = np.zeros(50, dtype=np.float32)
        dt2 = np.zeros(50, dtype=np.float32)
        dt3 = np.zeros(50, dtype=np.float32)
        deck_size = np.zeros(1, dtype=np.float32)
        scores = np.zeros(2, dtype=np.float32)
        phase = np.zeros(1, dtype=np.int32)
        played_card = np.zeros(50, dtype=np.float32)
        played_type = np.zeros(1, dtype=np.float32)
        s.encode_obs(player_id, hand, own_exp, opp_exp,
                     dt1, dt2, dt3, deck_size, scores, phase,
                     played_card, played_type)
        return LCObs(hand=hand, own_expeditions=own_exp,
                     opp_expeditions=opp_exp,
                     discard_top1=dt1, discard_top2=dt2, discard_top3=dt3,
                     deck_size=deck_size, scores=scores, phase=phase,
                     played_card=played_card, played_type=played_type)

    def encode_into(self, player_id: int, out: LCObs, idx: int) -> None:
        self._state.encode_obs(
            player_id,
            out.hand[idx], out.own_expeditions[idx], out.opp_expeditions[idx],
            out.discard_top1[idx], out.discard_top2[idx], out.discard_top3[idx],
            out.deck_size[idx], out.scores[idx], out.phase[idx],
            out.played_card[idx], out.played_type[idx])

    def _calculate_scores(self) -> np.ndarray:
        self._state.calc_scores()
        return self._state.scores.copy()

    def compute_scores(self) -> np.ndarray:
        if self._state.is_done:
            return self._state.scores.copy()
        self._state.calc_scores()
        return self._state.scores.copy()

    def game_metrics(self, player_idx: int) -> dict:
        s = self._state
        num_inv = 0
        num_played = 0
        open_lanes = 0
        for color in range(NUM_COLORS):
            n = s.exp_len[player_idx, color]
            if n == 0:
                continue
            open_lanes += 1
            num_played += n
            for j in range(n):
                if s.exp_vals[player_idx, color, j] == 0:
                    num_inv += 1
        return {
            "total_points": float(s.scores[player_idx]) if s.is_done else float(self.compute_scores()[player_idx]),
            "num_investment": num_inv,
            "num_played": num_played,
            "open_lanes": open_lanes,
            "cards_in_hand": int(s.hands[player_idx].sum()),
            "discard_draws": int(s.discard_draws[player_idx]),
        }
