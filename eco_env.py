"""
R-öko Card Game Environment for Reinforcement Learning.

Rules: eco_rule.md

Deck (per game, all player counts):
  19 single cards (value=1) per color × 4 colors = 76 cards
   3 double cards (value=2) per color × 4 colors = 12 cards
  Total: 88 recycling cards

Colors: 0=Glass, 1=Paper, 2=Plastic, 3=Tin
Types : 0=Single (value 1),  1=Double (value 2)

Action space (108 total, phase-dependent masking):
  Play actions    0..99  : color*25 + n_singles*5 + n_doubles
  Discard actions 100..107: 100 + color*2 + card_type
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_COLORS  = 4
NUM_TYPES   = 2            # 0=single, 1=double
CARD_VALUES = [1, 2]       # card value by type index

SINGLES_PER_COLOR = 19   # 19 × 4 colors = 76 single cards
DOUBLES_PER_COLOR = 3    #  3 × 4 colors = 12 double cards  (total deck: 88)

HAND_LIMIT          = 5
MIN_RECYCLE_VALUE   = 4    # recycling side must hit this to claim a factory card

# Factory stacks (top → bottom order: first element is taken first)
_STACK = {
    2: [0, 1, 2, -2, 4, 5],
    3: [0, 1, 2,  3, -2, 4, 5],
    4: [0, 1, 2,  3, -2, 4, 5],
    5: [0, 1, 2,  3,  3, -2, 4, 5],
}

MAX_ECO_SCORE = 50   # normalization upper bound

# Phases
PHASE_PLAY    = 0
PHASE_DISCARD = 1

# Action encoding
NUM_PLAY_ACTIONS    = NUM_COLORS * 5 * 5   # 100
NUM_DISCARD_ACTIONS = NUM_COLORS * NUM_TYPES  # 8
NUM_ACTIONS         = NUM_PLAY_ACTIONS + NUM_DISCARD_ACTIONS  # 108


# ── Action codec ──────────────────────────────────────────────────────────────

def encode_play(color: int, n_singles: int, n_doubles: int) -> int:
    return color * 25 + n_singles * 5 + n_doubles

def decode_play(action: int) -> Tuple[int, int, int]:
    color     = action // 25
    remainder = action % 25
    return color, remainder // 5, remainder % 5

def encode_discard(color: int, card_type: int) -> int:
    return NUM_PLAY_ACTIONS + color * NUM_TYPES + card_type

def decode_discard(action: int) -> Tuple[int, int]:
    idx = action - NUM_PLAY_ACTIONS
    return idx // NUM_TYPES, idx % NUM_TYPES


# ── Game state ────────────────────────────────────────────────────────────────

@dataclass
class EcoState:
    """Complete (perfect-information) R-öko game state."""

    # hands[p][c][t] = count of (color c, type t) cards held by player p
    hands: np.ndarray               # (num_players, NUM_COLORS, NUM_TYPES) int32

    # recycling_side[c][t] = count of cards on factory c's recycling side
    recycling_side: np.ndarray      # (NUM_COLORS, NUM_TYPES) int32

    # waste_side[f][c][t] = count of (color c, type t) cards on factory f's waste side
    waste_side: np.ndarray          # (NUM_COLORS, NUM_COLORS, NUM_TYPES) int32

    # factory_stacks[color] = remaining factory cards (list, first = top)
    factory_stacks: List[List[int]]

    # collected[p][c] = list of factory card values taken by player p in color c
    collected: List[List[List[int]]]

    # face-down penalty cards per player, tracked by composition (perfect information)
    # penalty_pile[p][c][t] = count of (color c, type t) cards in player p's penalty pile
    penalty_pile: np.ndarray        # (num_players, NUM_COLORS, NUM_TYPES) int32

    # discard pile composition (cleared recycling-side cards)
    discard_counts: np.ndarray      # (NUM_COLORS, NUM_TYPES) int32

    # draw pile as ordered list of (color, type); index -1 = top of pile
    draw_pile: List[Tuple[int, int]]

    current_player: int
    num_players: int
    phase: int                       # PHASE_PLAY or PHASE_DISCARD

    done: bool = False

    # Set True when a factory empties mid-turn; full done+scoring at turn end
    pending_done: bool = False

    # Factory color played this turn (needed for waste refill after discards)
    played_color: int = -1

    # Number of new waste cards to draw when the current turn fully ends
    pending_draw_count: int = 1


# ── Environment ───────────────────────────────────────────────────────────────

class EcoEnv:
    """Single-instance R-öko game environment (perfect information)."""

    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        assert 2 <= num_players <= 5, "num_players must be 2–5"
        self.num_players = num_players
        self.rng = np.random.default_rng(seed)
        self.state: Optional[EcoState] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> EcoState:
        draw_pile = self._build_deck()

        factory_stacks = [list(_STACK[self.num_players]) for _ in range(NUM_COLORS)]
        collected      = [[[] for _ in range(NUM_COLORS)] for _ in range(self.num_players)]

        hands         = np.zeros((self.num_players, NUM_COLORS, NUM_TYPES), dtype=np.int32)
        waste_side    = np.zeros((NUM_COLORS, NUM_COLORS, NUM_TYPES), dtype=np.int32)
        recycling_side = np.zeros((NUM_COLORS, NUM_TYPES), dtype=np.int32)
        discard_counts = np.zeros((NUM_COLORS, NUM_TYPES), dtype=np.int32)

        # One card per factory waste side
        for f in range(NUM_COLORS):
            self._draw_into(draw_pile, waste_side[f])

        # Deal 3 cards to each player
        for p in range(self.num_players):
            for _ in range(3):
                self._draw_into(draw_pile, hands[p])

        self.state = EcoState(
            hands=hands,
            recycling_side=recycling_side,
            waste_side=waste_side,
            factory_stacks=factory_stacks,
            collected=collected,
            penalty_pile=np.zeros((self.num_players, NUM_COLORS, NUM_TYPES), dtype=np.int32),
            discard_counts=discard_counts,
            draw_pile=draw_pile,
            current_player=0,
            num_players=self.num_players,
            phase=PHASE_PLAY,
        )
        return self.state

    def legal_actions(self, state: Optional[EcoState] = None) -> np.ndarray:
        """Return bool mask (NUM_ACTIONS,) of legal actions for the current player."""
        s = state or self.state
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        p    = s.current_player
        hand = s.hands[p]  # (NUM_COLORS, NUM_TYPES)

        if s.phase == PHASE_PLAY:
            for color in range(NUM_COLORS):
                max_s = int(hand[color, 0])
                max_d = int(hand[color, 1])
                for n_s in range(5):
                    if n_s > max_s:
                        continue
                    for n_d in range(5):
                        if n_d > max_d:
                            continue
                        if n_s + n_d == 0:
                            continue  # must play at least one card
                        mask[encode_play(color, n_s, n_d)] = True

        else:  # PHASE_DISCARD
            for color in range(NUM_COLORS):
                for t in range(NUM_TYPES):
                    if hand[color, t] > 0:
                        mask[encode_discard(color, t)] = True

        return mask

    def step(self, action: int):
        """
        Execute action.  Returns (state, rewards, done, info).
        rewards: (num_players,) float32.
        """
        s = self.state
        assert not s.done, "Game over – call reset()"
        assert self.legal_actions(s)[action], (
            f"Illegal action {action} in phase {s.phase} for player {s.current_player}"
        )

        if s.phase == PHASE_PLAY:
            rewards = self._step_play(action)
        else:
            rewards = self._step_discard(action)

        return s, rewards, s.done, {}

    def compute_scores(self, state: Optional[EcoState] = None) -> np.ndarray:
        """Compute current/final scores for all players. Returns (num_players,) float32."""
        s = state or self.state
        scores = np.zeros(s.num_players, dtype=np.float32)

        # Factory-card scoring: only count a color if player has >1 card of it
        for p in range(s.num_players):
            for c in range(NUM_COLORS):
                cards = s.collected[p][c]
                if len(cards) > 1:
                    scores[p] += sum(cards)

        # Penalty deduction (sum over color×type per player)
        penalty_counts = s.penalty_pile.sum(axis=(1, 2)).astype(np.float32)
        scores -= penalty_counts

        # Clean-player bonus
        players_with_penalty = int((penalty_counts > 0).sum())
        if players_with_penalty > 0:
            bonus_map = {2: 3, 3: 2, 4: 1, 5: 1}
            bonus = bonus_map.get(s.num_players, 1)
            for p in range(s.num_players):
                if penalty_counts[p] == 0:
                    scores[p] += bonus

        return scores

    def observe(self) -> dict:
        """Raw observation dict for current player."""
        s = self.state
        return {
            "hands":          s.hands.copy(),
            "recycling_side": s.recycling_side.copy(),
            "waste_side":     s.waste_side.copy(),
            "factory_stacks": [list(st) for st in s.factory_stacks],
            "collected":      [[list(col) for col in player] for player in s.collected],
            "penalty_pile":   s.penalty_pile.copy(),
            "draw_pile_size": len(s.draw_pile),
            "current_player": s.current_player,
            "phase":          s.phase,
            "legal_actions":  self.legal_actions(),
        }

    # ── Internal: step helpers ────────────────────────────────────────────────

    def _step_play(self, action: int) -> np.ndarray:
        s = self.state
        p = s.current_player
        color, n_s, n_d = decode_play(action)
        rewards = np.zeros(s.num_players, dtype=np.float32)

        # Remove cards from hand, add to recycling side
        s.hands[p, color, 0] -= n_s
        s.hands[p, color, 1] -= n_d
        s.recycling_side[color, 0] += n_s
        s.recycling_side[color, 1] += n_d

        # Total value on recycling side (existing + newly played)
        total = int(
            s.recycling_side[color, 0] * CARD_VALUES[0] +
            s.recycling_side[color, 1] * CARD_VALUES[1]
        )

        if total >= MIN_RECYCLE_VALUE:
            # Take top factory card (if any remain)
            if s.factory_stacks[color]:
                fcard = s.factory_stacks[color].pop(0)
                s.collected[p][color].append(fcard)
                # Immediate reward proportional to card value
                rewards[p] += fcard / MAX_ECO_SCORE
                if not s.factory_stacks[color]:
                    s.pending_done = True  # will finalise at end of turn

            # Clear recycling side → discard pile
            s.discard_counts[color, 0] += s.recycling_side[color, 0]
            s.discard_counts[color, 1] += s.recycling_side[color, 1]
            s.recycling_side[color]     = 0

            draw_count = 1   # recycling side cleared → draw 1
        else:
            # Recycling side NOT cleared; draw 1 + current side value
            draw_count = 1 + total

        # Take all waste cards from this factory
        s.hands[p] += s.waste_side[color]
        s.waste_side[color] = np.zeros((NUM_COLORS, NUM_TYPES), dtype=np.int32)

        # Save turn info for potential discard phase
        s.played_color       = color
        s.pending_draw_count = draw_count

        # Check hand limit
        if int(s.hands[p].sum()) > HAND_LIMIT:
            s.phase = PHASE_DISCARD
            # Refill happens after discarding finishes
        else:
            rewards += self._end_of_turn()

        return rewards

    def _step_discard(self, action: int) -> np.ndarray:
        s = self.state
        p = s.current_player
        rewards = np.zeros(s.num_players, dtype=np.float32)

        color, t = decode_discard(action)
        s.hands[p, color, t]        -= 1
        s.penalty_pile[p, color, t] += 1
        # Immediate penalty shaping (mirrors factory-card shaping in _step_play)
        rewards[p] -= 1.0 / MAX_ECO_SCORE

        if int(s.hands[p].sum()) <= HAND_LIMIT:
            rewards += self._end_of_turn()

        return rewards

    def _end_of_turn(self) -> np.ndarray:
        """Refill waste pile, advance player, check for game end."""
        s = self.state
        rewards = np.zeros(s.num_players, dtype=np.float32)

        # Refill waste side for the factory that was played
        self._refill_waste(s.played_color, s.pending_draw_count)

        if s.pending_done:
            # Game over — just flag it. Terminal reward is handled by the
            # single-player wrapper (SinglePlayerEcoEnv._terminal_return).
            # Do NOT emit score-based rewards here to avoid double-counting.
            s.done = True
        else:
            s.current_player = (s.current_player + 1) % s.num_players
            s.phase          = PHASE_PLAY

        return rewards

    # ── Internal: deck / draw helpers ─────────────────────────────────────────

    def _build_deck(self) -> List[Tuple[int, int]]:
        deck = []
        for c in range(NUM_COLORS):
            deck.extend([(c, 0)] * SINGLES_PER_COLOR)
            deck.extend([(c, 1)] * DOUBLES_PER_COLOR)
        indices = self.rng.permutation(len(deck))
        return [deck[i] for i in indices]

    def _draw_into(self, draw_pile: List, target: np.ndarray) -> bool:
        """Pop one card from draw_pile into target[color,type]. Returns True if drawn."""
        if not draw_pile:
            return False
        c, t = draw_pile.pop()
        target[c, t] += 1
        return True

    def _refill_waste(self, factory_color: int, count: int):
        """Draw `count` cards from deck into factory_color's waste side."""
        s = self.state
        for _ in range(count):
            if not s.draw_pile:
                self._reshuffle()
                if not s.draw_pile:
                    break
            self._draw_into(s.draw_pile, s.waste_side[factory_color])

    def _reshuffle(self):
        """Rebuild draw pile from discard pile."""
        s = self.state
        if s.discard_counts.sum() == 0:
            return
        deck = []
        for c in range(NUM_COLORS):
            for t in range(NUM_TYPES):
                deck.extend([(c, t)] * int(s.discard_counts[c, t]))
        indices = self.rng.permutation(len(deck))
        s.draw_pile     = [deck[i] for i in indices]
        s.discard_counts[:] = 0
