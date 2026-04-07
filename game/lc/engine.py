"""
Lost Cities — BaseGameEngine implementation.

2-player card game. 60 cards: 5 colors × 12 cards (3 investments + numbered 2–10).
Each turn: play a card (expedition or discard) then draw a card (deck or discard pile).
Scoring: for each expedition, (sum_values - penalty) × (1 + num_investments), +20 bonus if ≥8 cards.

Action encoding (600 total):
  flat = card_id * 12 + action_type * 6 + draw_source
  card_id ∈ [0, 50)  — 5 colors × 10 unique values
  action_type ∈ {0=expedition, 1=discard}
  draw_source ∈ [0, 6) — 0=deck, 1–5=discard piles by color
"""

from typing import NamedTuple
import numpy as np
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
#   Play actions  0..99:  card_id * 2 + action_type (expedition=0, discard=1)
#   Draw actions  100..105: 100 + draw_source (0=deck, 1-5=discard piles)
NUM_PLAY_ACTIONS = NUM_CARD_IDS * NUM_ACTION_TYPES  # 100
NUM_DRAW_ACTIONS = NUM_DRAW_SOURCES                  # 6
NUM_DECOMPOSED_ACTIONS = NUM_PLAY_ACTIONS + NUM_DRAW_ACTIONS  # 106

# Phases (for 3-phase decomposed action mode)
PHASE3_SELECT_CARD = 0   # choose which card from hand
PHASE3_ACTION_TYPE = 1   # expedition or discard
PHASE3_DRAW = 2          # choose draw source

# 3-phase decomposed action space (58 total, phase-dependent masking):
#   Card select  0..49:  card_id
#   Action type  50..51: 50 + action_type (0=expedition, 1=discard)
#   Draw source  52..57: 52 + draw_source (0=deck, 1-5=discard piles)
NUM_CARD_SELECT_ACTIONS = NUM_CARD_IDS               # 50
NUM_TYPE_ACTIONS = NUM_ACTION_TYPES                   # 2
NUM_3PHASE_DRAW_ACTIONS = NUM_DRAW_SOURCES            # 6
NUM_3PHASE_ACTIONS = NUM_CARD_SELECT_ACTIONS + NUM_TYPE_ACTIONS + NUM_3PHASE_DRAW_ACTIONS  # 58

# Card values: index 0 = investment (value 0), index k (k>=1) = value k+1
def _card_value(value_idx: int) -> int:
    return 0 if value_idx == 0 else value_idx + 1

CARD_VALUES = [_card_value(i) for i in range(NUM_UNIQUE_VALUES)]
# How many copies of each unique card exist in the deck
# Investment (value_idx=0): 3 copies; numbered (value_idx 1-9): 1 copy each
CARD_COPIES = [3] + [1] * 9


# ── Action encoding ────────────────────────────────────────────────────────

def encode_action(card_id: int, action_type: int, draw_source: int) -> int:
    return card_id * 12 + action_type * 6 + draw_source

def decode_action(flat: int) -> tuple[int, int, int]:
    draw_source = flat % 6
    action_type = (flat // 6) % 2
    card_id = flat // 12
    return card_id, action_type, draw_source

def card_id_to_color_value(card_id: int) -> tuple[int, int]:
    """card_id -> (color_idx, value_idx)"""
    return card_id // NUM_UNIQUE_VALUES, card_id % NUM_UNIQUE_VALUES

def color_value_to_card_id(color_idx: int, value_idx: int) -> int:
    return color_idx * NUM_UNIQUE_VALUES + value_idx


# ── Decomposed action encoding ────────────────────────────────────────────

def encode_play_action(card_id: int, action_type: int) -> int:
    """Encode play-phase action: card_id * 2 + action_type."""
    return card_id * 2 + action_type

def decode_play_action(action: int) -> tuple[int, int]:
    """Decode play-phase action → (card_id, action_type)."""
    return action // 2, action % 2

def encode_draw_action(draw_source: int) -> int:
    """Encode draw-phase action: NUM_PLAY_ACTIONS + draw_source."""
    return NUM_PLAY_ACTIONS + draw_source

def decode_draw_action(action: int) -> int:
    """Decode draw-phase action → draw_source."""
    return action - NUM_PLAY_ACTIONS


# ── 3-phase decomposed action encoding ───────────────────────────────────

def encode_card_select_action(card_id: int) -> int:
    """Encode card-select phase action."""
    return card_id

def decode_card_select_action(action: int) -> int:
    """Decode card-select phase action → card_id."""
    return action

def encode_type_action(action_type: int) -> int:
    """Encode action-type phase action: NUM_CARD_SELECT_ACTIONS + action_type."""
    return NUM_CARD_SELECT_ACTIONS + action_type

def decode_type_action(action: int) -> int:
    """Decode action-type phase action → action_type."""
    return action - NUM_CARD_SELECT_ACTIONS

def encode_3phase_draw_action(draw_source: int) -> int:
    """Encode 3-phase draw action: NUM_CARD_SELECT_ACTIONS + NUM_TYPE_ACTIONS + draw_source."""
    return NUM_CARD_SELECT_ACTIONS + NUM_TYPE_ACTIONS + draw_source

def decode_3phase_draw_action(action: int) -> int:
    """Decode 3-phase draw action → draw_source."""
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
    phase: np.ndarray             # (1,) int32: 0=play, 1=draw (2-phase); 0=select, 1=type, 2=draw (3-phase)
    played_card: np.ndarray       # (NUM_CARD_IDS,) one-hot of card selected/played
    played_type: np.ndarray       # (1,) 0=expedition, 1=discard (-1 when not yet chosen)

def float_dim(decompose_actions: bool = False, three_phase: bool = False) -> int:
    """Total float feature dimension (excludes phase, which is embedded)."""
    base = 50 * 6 + 1 + 2  # 303
    if decompose_actions or three_phase:
        base += NUM_CARD_IDS + 1  # played_card (50) + played_type (1)
    return base


# ── Engine ──────────────────────────────────────────────────────────────────

class LCEngine(BaseGameEngine[LCObs]):
    """Lost Cities game engine for 2 players."""

    SCORE_DIFF_NORM = 30.0  # normalization for score-difference reward

    def __init__(self, rng: np.random.Generator,
                 new_color_penalty: int = DEFAULT_NEW_COLOR_PENALTY,
                 score_diff_reward: bool = False,
                 zero_one_reward: bool = False,
                 max_lanes: int = NUM_COLORS,
                 decompose_actions: bool = False,
                 three_phase: bool = False,
                 max_discard_draws: int = 0,
                 raw_score_reward: bool = False,
                 dense_reward: bool = False):
        super().__init__(rng)
        self._new_color_penalty = new_color_penalty
        self._score_diff_reward = score_diff_reward
        self._zero_one_reward = zero_one_reward
        self._raw_score_reward = raw_score_reward
        self._dense_reward = dense_reward
        self._max_lanes = max_lanes
        self._decompose_actions = decompose_actions
        self._three_phase = three_phase
        self._max_discard_draws = max_discard_draws
        self._prev_scores = np.zeros(NUM_PLAYERS, dtype=np.float32)
        # State arrays — allocated once, reset each game
        self._hands: list[np.ndarray] = [np.zeros(NUM_CARD_IDS, dtype=np.int8) for _ in range(NUM_PLAYERS)]
        self._expeditions: list[list[list[int]]] = [[[] for _ in range(NUM_COLORS)] for _ in range(NUM_PLAYERS)]
        self._discard_piles: list[list[int]] = [[] for _ in range(NUM_COLORS)]
        self._deck: list[int] = []  # list of card_ids
        self._current_player = 0
        self._done = False
        self._scores = np.zeros(NUM_PLAYERS, dtype=np.float32)
        self._discard_draws = [0, 0]  # count of draws from discard piles per player
        # Decomposed action state (phase tracking)
        self._phase = PHASE3_SELECT_CARD if three_phase else PHASE_PLAY
        self._played_color = -1       # color of card selected/played
        self._played_val_idx = -1     # value index of card selected/played
        self._played_card_id = -1     # card_id of card selected (3-phase)
        self._played_action_type = -1  # 0=expedition, 1=discard

    def _assign_terminal_rewards(self, rewards: list[float]) -> None:
        """Fill terminal rewards based on reward mode."""
        if self._raw_score_reward:
            for i in range(NUM_PLAYERS):
                rewards[i] = self._scores[i] / self.SCORE_DIFF_NORM
            return
        if self._score_diff_reward:
            for i in range(NUM_PLAYERS):
                opp = 1 - i
                rewards[i] = (self._scores[i] - self._scores[opp]) / self.SCORE_DIFF_NORM
        elif self._zero_one_reward:
            best = float(self._scores.max())
            for i in range(NUM_PLAYERS):
                rewards[i] = 1.0 if self._scores[i] >= best else 0.0
        else:
            best = float(self._scores.max())
            worst = float(self._scores.min())
            for i in range(NUM_PLAYERS):
                if self._scores[i] >= best:
                    rewards[i] = 1.0
                elif self._scores[i] <= worst:
                    rewards[i] = -1.0

    def _reset(self) -> None:
        # Build and shuffle deck
        deck = []
        for color in range(NUM_COLORS):
            for val_idx in range(NUM_UNIQUE_VALUES):
                for _ in range(CARD_COPIES[val_idx]):
                    deck.append(color_value_to_card_id(color, val_idx))
        self.rng.shuffle(deck)
        self._deck = list(deck)

        # Reset state
        for p in range(NUM_PLAYERS):
            self._hands[p][:] = 0
            self._expeditions[p] = [[] for _ in range(NUM_COLORS)]
        self._discard_piles = [[] for _ in range(NUM_COLORS)]
        self._current_player = 0
        self._done = False
        self._scores[:] = 0
        self._discard_draws = [0, 0]
        self._prev_scores[:] = 0
        self._phase = PHASE3_SELECT_CARD if self._three_phase else PHASE_PLAY
        self._played_color = -1
        self._played_val_idx = -1
        self._played_card_id = -1
        self._played_action_type = -1

        # Deal hands
        for _ in range(HAND_SIZE):
            for p in range(NUM_PLAYERS):
                card_id = self._deck.pop()
                self._hands[p][card_id] += 1

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def done(self) -> bool:
        return self._done

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

    # ── Step ────────────────────────────────────────────────────────────

    def step(self, action: int) -> tuple[float, ...]:
        if self._three_phase:
            return self._step_3phase(action)
        if self._decompose_actions:
            return self._step_decomposed(action)
        return self._step_flat(action)

    def _step_flat(self, action: int) -> tuple[float, ...]:
        """Original flat action step (600 actions)."""
        rewards = [0.0] * NUM_PLAYERS
        card_id, action_type, draw_source = decode_action(action)
        p = self._current_player
        hand = self._hands[p]
        color, val_idx = card_id_to_color_value(card_id)

        assert hand[card_id] > 0, f"Player {p} doesn't have card {card_id}"

        # Play phase
        hand[card_id] -= 1
        if action_type == 0:  # expedition
            self._expeditions[p][color].append(val_idx)
        else:  # discard
            self._discard_piles[color].append(card_id)

        # Draw phase
        if draw_source == 0:  # deck
            drawn = self._deck.pop()
            hand[drawn] += 1
        else:  # discard pile (source 1-5 -> color 0-4)
            pile_color = draw_source - 1
            drawn = self._discard_piles[pile_color].pop()
            hand[drawn] += 1
            self._discard_draws[p] += 1

        # Check game end
        if len(self._deck) == 0:
            self._done = True
            self._scores = self._calculate_scores()
            if self._dense_reward:
                for i in range(NUM_PLAYERS):
                    rewards[i] = (self._scores[i] - self._prev_scores[i]) / self.SCORE_DIFF_NORM
            else:
                self._assign_terminal_rewards(rewards)
        else:
            if self._dense_reward:
                cur_scores = self._calculate_scores()
                rewards[p] = (cur_scores[p] - self._prev_scores[p]) / self.SCORE_DIFF_NORM
                self._prev_scores[:] = cur_scores
            self._current_player = 1 - p

        return tuple(rewards)

    def _step_decomposed(self, action: int) -> tuple[float, ...]:
        """Decomposed two-phase step (106 actions)."""
        if self._phase == PHASE_PLAY:
            return self._step_play_phase(action)
        else:
            return self._step_draw_phase(action)

    def _step_play_phase(self, action: int) -> tuple[float, ...]:
        """Phase 1: play a card (expedition or discard). No reward, stays on same player."""
        assert action < NUM_PLAY_ACTIONS, f"Expected play action 0..{NUM_PLAY_ACTIONS-1}, got {action}"
        card_id, action_type = decode_play_action(action)
        p = self._current_player
        hand = self._hands[p]
        color, val_idx = card_id_to_color_value(card_id)

        assert hand[card_id] > 0, f"Player {p} doesn't have card {card_id}"

        hand[card_id] -= 1
        if action_type == 0:  # expedition
            self._expeditions[p][color].append(val_idx)
        else:  # discard
            self._discard_piles[color].append(card_id)

        # Remember what was played (for draw-phase masking + obs)
        self._played_color = color
        self._played_val_idx = val_idx
        self._played_action_type = action_type
        self._phase = PHASE_DRAW
        return tuple([0.0] * NUM_PLAYERS)

    def _step_draw_phase(self, action: int) -> tuple[float, ...]:
        """Phase 2: draw a card. May produce terminal reward. Switches player."""
        assert action >= NUM_PLAY_ACTIONS, f"Expected draw action {NUM_PLAY_ACTIONS}..{NUM_DECOMPOSED_ACTIONS-1}, got {action}"
        draw_source = decode_draw_action(action)
        p = self._current_player
        hand = self._hands[p]
        rewards = [0.0] * NUM_PLAYERS

        if draw_source == 0:  # deck
            drawn = self._deck.pop()
            hand[drawn] += 1
        else:  # discard pile
            pile_color = draw_source - 1
            drawn = self._discard_piles[pile_color].pop()
            hand[drawn] += 1
            self._discard_draws[p] += 1

        # Check game end
        if len(self._deck) == 0:
            self._done = True
            self._scores = self._calculate_scores()
            self._assign_terminal_rewards(rewards)
        else:
            self._current_player = 1 - p

        self._phase = PHASE_PLAY
        return tuple(rewards)

    # ── 3-phase step ───────────────────────────────────────────────────

    def _step_3phase(self, action: int) -> tuple[float, ...]:
        """3-phase decomposed step (58 actions)."""
        if self._phase == PHASE3_SELECT_CARD:
            return self._step_select_card(action)
        elif self._phase == PHASE3_ACTION_TYPE:
            return self._step_action_type(action)
        else:
            return self._step_3phase_draw(action)

    def _step_select_card(self, action: int) -> tuple[float, ...]:
        """Phase 1: select a card from hand. No reward, stays on same player."""
        assert action < NUM_CARD_SELECT_ACTIONS, f"Expected card select 0..{NUM_CARD_SELECT_ACTIONS-1}, got {action}"
        card_id = decode_card_select_action(action)
        p = self._current_player
        assert self._hands[p][card_id] > 0, f"Player {p} doesn't have card {card_id}"

        color, val_idx = card_id_to_color_value(card_id)
        self._played_card_id = card_id
        self._played_color = color
        self._played_val_idx = val_idx
        self._phase = PHASE3_ACTION_TYPE
        return tuple([0.0] * NUM_PLAYERS)

    def _step_action_type(self, action: int) -> tuple[float, ...]:
        """Phase 2: expedition or discard the selected card. No reward, stays on same player."""
        assert NUM_CARD_SELECT_ACTIONS <= action < NUM_CARD_SELECT_ACTIONS + NUM_TYPE_ACTIONS, \
            f"Expected type action {NUM_CARD_SELECT_ACTIONS}..{NUM_CARD_SELECT_ACTIONS + NUM_TYPE_ACTIONS - 1}, got {action}"
        action_type = decode_type_action(action)
        p = self._current_player
        card_id = self._played_card_id
        color = self._played_color
        val_idx = self._played_val_idx

        # Now actually play the card
        self._hands[p][card_id] -= 1
        if action_type == 0:  # expedition
            self._expeditions[p][color].append(val_idx)
        else:  # discard
            self._discard_piles[color].append(card_id)

        self._played_action_type = action_type
        self._phase = PHASE3_DRAW
        return tuple([0.0] * NUM_PLAYERS)

    def _step_3phase_draw(self, action: int) -> tuple[float, ...]:
        """Phase 3: draw a card. May produce terminal reward. Switches player."""
        assert action >= NUM_CARD_SELECT_ACTIONS + NUM_TYPE_ACTIONS, \
            f"Expected draw action {NUM_CARD_SELECT_ACTIONS + NUM_TYPE_ACTIONS}..{NUM_3PHASE_ACTIONS-1}, got {action}"
        draw_source = decode_3phase_draw_action(action)
        p = self._current_player
        hand = self._hands[p]
        rewards = [0.0] * NUM_PLAYERS

        if draw_source == 0:  # deck
            drawn = self._deck.pop()
            hand[drawn] += 1
        else:  # discard pile
            pile_color = draw_source - 1
            drawn = self._discard_piles[pile_color].pop()
            hand[drawn] += 1
            self._discard_draws[p] += 1

        # Check game end
        if len(self._deck) == 0:
            self._done = True
            self._scores = self._calculate_scores()
            self._assign_terminal_rewards(rewards)
        else:
            self._current_player = 1 - p

        self._phase = PHASE3_SELECT_CARD
        self._played_card_id = -1
        self._played_color = -1
        self._played_val_idx = -1
        self._played_action_type = -1
        return tuple(rewards)

    # ── Legal actions ───────────────────────────────────────────────────

    def _is_valid_expedition_play(self, player: int, card_id: int) -> bool:
        color, val_idx = card_id_to_color_value(card_id)
        exp = self._expeditions[player][color]
        if not exp:
            return True
        last_val = exp[-1]
        if val_idx == 0:  # investment
            num_inv = sum(1 for v in exp if v == 0)
            return num_inv < 3 and last_val == 0
        else:  # numbered
            return val_idx > last_val

    def legal_actions(self) -> np.ndarray:
        if self._three_phase:
            return self._legal_actions_3phase()
        if self._decompose_actions:
            return self._legal_actions_decomposed()
        return self._legal_actions_flat()

    def _discard_draws_blocked(self) -> bool:
        """True if discard draws should be masked (limit reached)."""
        if self._max_discard_draws <= 0:
            return False
        return (self._discard_draws[0] + self._discard_draws[1]) >= self._max_discard_draws

    def _legal_actions_flat(self) -> np.ndarray:
        """Original flat legal actions (600-dim)."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        p = self._current_player
        hand = self._hands[p]
        block_ddraw = self._discard_draws_blocked()

        # Count open lanes for lane limit check
        open_lanes = sum(1 for c in range(NUM_COLORS) if self._expeditions[p][c])

        for card_id in range(NUM_CARD_IDS):
            if hand[card_id] <= 0:
                continue
            color, val_idx = card_id_to_color_value(card_id)

            # Expedition play (block opening a new lane if at max)
            can_open_new = open_lanes < self._max_lanes
            lane_is_open = len(self._expeditions[p][color]) > 0
            if self._is_valid_expedition_play(p, card_id) and (lane_is_open or can_open_new):
                for ds in range(NUM_DRAW_SOURCES):
                    if ds == 0:
                        if len(self._deck) > 0:
                            mask[encode_action(card_id, 0, ds)] = True
                    else:
                        if block_ddraw:
                            continue
                        pile_color = ds - 1
                        if len(self._discard_piles[pile_color]) > 0:
                            mask[encode_action(card_id, 0, ds)] = True

            # Discard
            for ds in range(NUM_DRAW_SOURCES):
                if ds == 0:
                    if len(self._deck) > 0:
                        mask[encode_action(card_id, 1, ds)] = True
                else:
                    if block_ddraw:
                        continue
                    pile_color = ds - 1
                    if pile_color == color:
                        continue
                    if len(self._discard_piles[pile_color]) > 0:
                        mask[encode_action(card_id, 1, ds)] = True

        return mask

    def _legal_actions_decomposed(self) -> np.ndarray:
        """Decomposed legal actions (106-dim), phase-dependent."""
        mask = np.zeros(NUM_DECOMPOSED_ACTIONS, dtype=bool)
        p = self._current_player

        if self._phase == PHASE_PLAY:
            hand = self._hands[p]
            open_lanes = sum(1 for c in range(NUM_COLORS) if self._expeditions[p][c])

            for card_id in range(NUM_CARD_IDS):
                if hand[card_id] <= 0:
                    continue
                color, val_idx = card_id_to_color_value(card_id)

                # Expedition play
                can_open_new = open_lanes < self._max_lanes
                lane_is_open = len(self._expeditions[p][color]) > 0
                if self._is_valid_expedition_play(p, card_id) and (lane_is_open or can_open_new):
                    mask[encode_play_action(card_id, 0)] = True

                # Discard (always legal for cards in hand)
                mask[encode_play_action(card_id, 1)] = True

        else:  # PHASE_DRAW
            # Deck draw
            if len(self._deck) > 0:
                mask[encode_draw_action(0)] = True

            # Discard pile draws (blocked if limit reached)
            if not self._discard_draws_blocked():
                for c in range(NUM_COLORS):
                    # Can't draw from the pile you just discarded to
                    if self._played_action_type == 1 and c == self._played_color:
                        continue
                    if self._discard_piles[c]:
                        mask[encode_draw_action(c + 1)] = True

        return mask

    def _legal_actions_3phase(self) -> np.ndarray:
        """3-phase legal actions (58-dim), phase-dependent."""
        mask = np.zeros(NUM_3PHASE_ACTIONS, dtype=bool)
        p = self._current_player

        if self._phase == PHASE3_SELECT_CARD:
            hand = self._hands[p]
            for card_id in range(NUM_CARD_IDS):
                if hand[card_id] > 0:
                    mask[encode_card_select_action(card_id)] = True

        elif self._phase == PHASE3_ACTION_TYPE:
            card_id = self._played_card_id
            # Expedition: check validity + lane limit
            open_lanes = sum(1 for c in range(NUM_COLORS) if self._expeditions[p][c])
            can_open_new = open_lanes < self._max_lanes
            color = self._played_color
            lane_is_open = len(self._expeditions[p][color]) > 0
            if self._is_valid_expedition_play(p, card_id) and (lane_is_open or can_open_new):
                mask[encode_type_action(0)] = True
            # Discard: always legal
            mask[encode_type_action(1)] = True

        else:  # PHASE3_DRAW
            # Deck draw
            if len(self._deck) > 0:
                mask[encode_3phase_draw_action(0)] = True
            # Discard pile draws (blocked if limit reached)
            if not self._discard_draws_blocked():
                for c in range(NUM_COLORS):
                    if self._played_action_type == 1 and c == self._played_color:
                        continue
                    if self._discard_piles[c]:
                        mask[encode_3phase_draw_action(c + 1)] = True

        return mask

    # ── Observation ─────────────────────────────────────────────────────

    def encode(self, player_id: int) -> LCObs:
        opp = 1 - player_id
        hand = self._hands[player_id].astype(np.float32) / 3.0
        own_exp = np.zeros(NUM_CARD_IDS, dtype=np.float32)
        opp_exp = np.zeros(NUM_CARD_IDS, dtype=np.float32)

        for color in range(NUM_COLORS):
            for val_idx in self._expeditions[player_id][color]:
                own_exp[color_value_to_card_id(color, val_idx)] += 1.0 / 3.0
            for val_idx in self._expeditions[opp][color]:
                opp_exp[color_value_to_card_id(color, val_idx)] += 1.0 / 3.0

        # Discard piles: encode top 3 cards
        dt1 = np.zeros(NUM_CARD_IDS, dtype=np.float32)
        dt2 = np.zeros(NUM_CARD_IDS, dtype=np.float32)
        dt3 = np.zeros(NUM_CARD_IDS, dtype=np.float32)
        for color in range(NUM_COLORS):
            pile = self._discard_piles[color]
            if len(pile) >= 1:
                dt1[pile[-1]] = 1.0
            if len(pile) >= 2:
                dt2[pile[-2]] = 1.0
            if len(pile) >= 3:
                dt3[pile[-3]] = 1.0

        deck_size = np.array([len(self._deck) / 44.0], dtype=np.float32)

        # Current scores, rotated so own score is first
        cur_scores = self._calculate_scores()
        scores = np.array([cur_scores[player_id] / 100.0,
                           cur_scores[opp] / 100.0], dtype=np.float32)

        phase = np.array([self._phase], dtype=np.int32)

        # Played card info (for decomposed/3-phase modes)
        played_card = np.zeros(NUM_CARD_IDS, dtype=np.float32)
        played_type = np.array([-1.0], dtype=np.float32)
        if self._decompose_actions and self._phase == PHASE_DRAW and self._played_color >= 0:
            played_card[color_value_to_card_id(self._played_color, self._played_val_idx)] = 1.0
            played_type[0] = float(self._played_action_type)
        elif self._three_phase and self._played_color >= 0:
            # In action_type phase: show selected card, type not yet chosen
            # In draw phase: show selected card + action type
            played_card[color_value_to_card_id(self._played_color, self._played_val_idx)] = 1.0
            if self._played_action_type >= 0:
                played_type[0] = float(self._played_action_type)

        return LCObs(
            hand=hand,
            own_expeditions=own_exp,
            opp_expeditions=opp_exp,
            discard_top1=dt1,
            discard_top2=dt2,
            discard_top3=dt3,
            deck_size=deck_size,
            scores=scores,
            phase=phase,
            played_card=played_card,
            played_type=played_type,
        )

    # ── Scoring ─────────────────────────────────────────────────────────

    def _calculate_scores(self) -> np.ndarray:
        scores = np.zeros(NUM_PLAYERS, dtype=np.float32)
        for p in range(NUM_PLAYERS):
            total = 0
            for color in range(NUM_COLORS):
                exp = self._expeditions[p][color]
                if not exp:
                    continue
                base_value = sum(CARD_VALUES[v] for v in exp)
                num_inv = sum(1 for v in exp if v == 0)
                multiplier = num_inv + 1
                exp_score = (base_value - self._new_color_penalty) * multiplier
                if len(exp) >= 8:
                    exp_score += EIGHT_CARD_BONUS
                total += exp_score
            scores[p] = total
        return scores

    def compute_scores(self) -> np.ndarray:
        if self._done:
            return self._scores.copy()
        return self._calculate_scores()

    def game_metrics(self, player_idx: int) -> dict:
        """Game metrics for a specific player (call after game ends)."""
        exp_list = self._expeditions[player_idx]
        num_investment = 0
        num_played = 0
        open_lanes = 0
        per_color_scores = []

        for color in range(NUM_COLORS):
            exp = exp_list[color]
            if not exp:
                continue
            open_lanes += 1
            num_played += len(exp)
            inv = sum(1 for v in exp if v == 0)
            num_investment += inv
            base_value = sum(CARD_VALUES[v] for v in exp)
            multiplier = inv + 1
            score = (base_value - self._new_color_penalty) * multiplier
            if len(exp) >= 8:
                score += EIGHT_CARD_BONUS
            per_color_scores.append(score)

        return {
            "total_points": float(self._scores[player_idx]) if self._done else float(self._calculate_scores()[player_idx]),
            "num_investment": num_investment,
            "num_played": num_played,
            "open_lanes": open_lanes,
            "cards_in_hand": int(self._hands[player_idx].sum()),
            "discard_draws": self._discard_draws[player_idx],
        }
