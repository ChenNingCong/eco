"""
R-Öko game engine — implements BaseGameEngine for the R-Öko card game.

Rules: see eco_rule.md

Deck (88 cards):
  19 single cards (value=1) per color x 4 colors = 76
   3 double cards (value=2) per color x 4 colors = 12

Colors: 0=Glass, 1=Paper, 2=Plastic, 3=Tin
Types:  0=Single (value 1), 1=Double (value 2)

Action space (108 total, phase-dependent masking):
  Play actions    0..99:   color*25 + n_singles*5 + n_doubles
  Discard actions 100..107: 100 + color*2 + card_type
"""

import numpy as np
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass

from abstract import BaseGameEngine


# ── Constants ────────────────────────────────────────────────────────────────

NUM_COLORS = 4
NUM_TYPES = 2              # 0=single, 1=double
CARD_VALUES = [1, 2]       # card value by type index

SINGLES_PER_COLOR = 19     # 19 x 4 colors = 76 single cards
DOUBLES_PER_COLOR = 3      #  3 x 4 colors = 12 double cards (total deck: 88)
TOTAL_DECK = NUM_COLORS * (SINGLES_PER_COLOR + DOUBLES_PER_COLOR)  # 88

HAND_LIMIT = 5
MIN_RECYCLE_VALUE = 4      # recycling side must hit this to claim factory card

# Factory stacks (top -> bottom: first element is taken first)
STACK_BY_PLAYERS = {
    2: [0, 1, 2, -2, 4, 5],
    3: [0, 1, 2, 3, -2, 4, 5],
    4: [0, 1, 2, 3, -2, 4, 5],
    5: [0, 1, 2, 3, 3, -2, 4, 5],
}

MAX_ECO_SCORE = 50         # normalisation upper bound

# Phases
PHASE_PLAY = 0
PHASE_DISCARD = 1

# Action encoding
NUM_PLAY_ACTIONS = NUM_COLORS * 5 * 5      # 100
NUM_DISCARD_ACTIONS = NUM_COLORS * NUM_TYPES  # 8
NUM_ACTIONS = NUM_PLAY_ACTIONS + NUM_DISCARD_ACTIONS  # 108


# ── Action codec ─────────────────────────────────────────────────────────────

def encode_play(color: int, n_singles: int, n_doubles: int) -> int:
    return color * 25 + n_singles * 5 + n_doubles

def decode_play(action: int) -> Tuple[int, int, int]:
    color = action // 25
    remainder = action % 25
    return color, remainder // 5, remainder % 5

def encode_discard(color: int, card_type: int) -> int:
    return NUM_PLAY_ACTIONS + color * NUM_TYPES + card_type

def decode_discard(action: int) -> Tuple[int, int]:
    idx = action - NUM_PLAY_ACTIONS
    return idx // NUM_TYPES, idx % NUM_TYPES


# ── Observation ──────────────────────────────────────────────────────────────

class RÖkoObs(NamedTuple):
    """Observation for R-Öko, rotated so the observing player is at index 0."""
    # Embedding index fields (int32)
    current_player: np.ndarray    # (1,) — always 0 (relative seat)
    phase: np.ndarray             # (1,)

    # Continuous fields (float32)
    hands: np.ndarray             # (num_players * NUM_COLORS * NUM_TYPES,)
    recycling_side: np.ndarray    # (NUM_COLORS * NUM_TYPES,)
    waste_side: np.ndarray        # (NUM_COLORS * NUM_COLORS * NUM_TYPES,)
    factory_stacks: np.ndarray    # (NUM_COLORS * stack_size,)
    collected: np.ndarray         # (num_players * NUM_COLORS * 2,)
    penalty_pile: np.ndarray      # (num_players * NUM_COLORS * NUM_TYPES,)
    scores: np.ndarray            # (num_players,)
    draw_pile_size: np.ndarray    # (1,)
    draw_pile_comp: np.ndarray    # (NUM_COLORS * NUM_TYPES,)


def float_dim(num_players: int = 2) -> int:
    """Total float feature dimension for the observation."""
    stack_size = len(STACK_BY_PLAYERS[num_players])
    return (
        num_players * NUM_COLORS * NUM_TYPES   # hands
        + NUM_COLORS * NUM_TYPES               # recycling_side
        + NUM_COLORS * NUM_COLORS * NUM_TYPES  # waste_side
        + NUM_COLORS * stack_size              # factory_stacks
        + num_players * NUM_COLORS * 2         # collected
        + num_players * NUM_COLORS * NUM_TYPES # penalty_pile
        + num_players                          # scores
        + 1                                    # draw_pile_size
        + NUM_COLORS * NUM_TYPES               # draw_pile_comp
    )


# ── Game state ───────────────────────────────────────────────────────────────

@dataclass
class EcoState:
    """Complete (perfect-information) R-Öko game state."""

    hands: np.ndarray               # (num_players, NUM_COLORS, NUM_TYPES) int32
    recycling_side: np.ndarray      # (NUM_COLORS, NUM_TYPES) int32
    waste_side: np.ndarray          # (NUM_COLORS, NUM_COLORS, NUM_TYPES) int32
    factory_stacks: List[List[int]]
    collected: List[List[List[int]]]
    penalty_pile: np.ndarray        # (num_players, NUM_COLORS, NUM_TYPES) int32
    discard_counts: np.ndarray      # (NUM_COLORS, NUM_TYPES) int32
    draw_pile: List[Tuple[int, int]]

    current_player: int
    phase: int                      # PHASE_PLAY or PHASE_DISCARD
    done: bool = False
    pending_done: bool = False
    played_color: int = -1
    pending_draw_count: int = 1


# ── Engine ───────────────────────────────────────────────────────────────────

class RÖkoEngine(BaseGameEngine[RÖkoObs]):
    """
    R-Öko card game engine.

    Parameters
    ----------
    key          : Key for RNG
    num_players  : 2-5
    """

    def __init__(self, rng: np.random.Generator, num_players: int = 2):
        super().__init__(rng)
        assert 2 <= num_players <= 5, "num_players must be 2-5"
        self._num_players = num_players
        self._stack_size = len(STACK_BY_PLAYERS[num_players])
        self.state: EcoState = None  # type: ignore

    # ── BaseGameEngine interface ─────────────────────────────────────────

    def _reset(self) -> None:
        draw_pile = self._build_deck(self.rng)

        n = self._num_players
        factory_stacks = [list(STACK_BY_PLAYERS[n]) for _ in range(NUM_COLORS)]
        collected = [[[] for _ in range(NUM_COLORS)] for _ in range(n)]

        hands = np.zeros((n, NUM_COLORS, NUM_TYPES), dtype=np.int32)
        waste_side = np.zeros((NUM_COLORS, NUM_COLORS, NUM_TYPES), dtype=np.int32)
        recycling_side = np.zeros((NUM_COLORS, NUM_TYPES), dtype=np.int32)
        discard_counts = np.zeros((NUM_COLORS, NUM_TYPES), dtype=np.int32)

        # One card per factory waste side
        for f in range(NUM_COLORS):
            self._draw_into(draw_pile, waste_side[f])

        # Deal 3 cards to each player
        for p in range(n):
            for _ in range(3):
                self._draw_into(draw_pile, hands[p])

        self.state = EcoState(
            hands=hands,
            recycling_side=recycling_side,
            waste_side=waste_side,
            factory_stacks=factory_stacks,
            collected=collected,
            penalty_pile=np.zeros((n, NUM_COLORS, NUM_TYPES), dtype=np.int32),
            discard_counts=discard_counts,
            draw_pile=draw_pile,
            current_player=0,
            phase=PHASE_PLAY,
        )

    def step(self, action: int) -> tuple[float, ...]:
        s = self.state
        assert not s.done, "Game over - call reset()"
        assert self.legal_actions()[action], (
            f"Illegal action {action} in phase {s.phase} "
            f"for player {s.current_player}"
        )

        if s.phase == PHASE_PLAY:
            rewards = self._step_play(action)
        else:
            rewards = self._step_discard(action)

        return tuple(float(r) for r in rewards)

    def legal_actions(self) -> np.ndarray:
        s = self.state
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        p = s.current_player
        hand = s.hands[p]

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
                            continue
                        mask[encode_play(color, n_s, n_d)] = True
        else:
            for color in range(NUM_COLORS):
                for t in range(NUM_TYPES):
                    if hand[color, t] > 0:
                        mask[encode_discard(color, t)] = True

        return mask

    def encode(self, player_id: int) -> RÖkoObs:
        s = self.state
        np_ = self._num_players

        # Rotate player-indexed arrays so player_id is at index 0
        perm = [(player_id + i) % np_ for i in range(np_)]

        # Factory stacks: consumed slots = -1.0, active slots /= 5
        factory_slots = np.full(
            (NUM_COLORS, self._stack_size), -1.0, dtype=np.float32
        )
        for c in range(NUM_COLORS):
            remaining = s.factory_stacks[c]
            n_consumed = self._stack_size - len(remaining)
            for i, v in enumerate(remaining):
                factory_slots[c, n_consumed + i] = v / 5.0

        # Hands normalised, rotated
        hands_flat = (
            s.hands[perm] / SINGLES_PER_COLOR
        ).astype(np.float32).flatten()

        # Recycling side normalised (shared)
        rec_flat = (
            s.recycling_side / SINGLES_PER_COLOR
        ).astype(np.float32).flatten()

        # Waste side normalised (shared)
        waste_flat = (
            s.waste_side / SINGLES_PER_COLOR
        ).astype(np.float32).flatten()

        # Collected: [scoring_flag, value_sum/MAX] per (player, color), rotated
        coll = np.zeros((np_, NUM_COLORS, 2), dtype=np.float32)
        for idx, pp in enumerate(perm):
            for c in range(NUM_COLORS):
                cards = s.collected[pp][c]
                coll[idx, c, 0] = 1.0 if len(cards) > 1 else 0.0
                coll[idx, c, 1] = sum(cards) / MAX_ECO_SCORE

        # Penalty pile normalised, rotated
        pen_norm = (
            s.penalty_pile[perm] / SINGLES_PER_COLOR
        ).astype(np.float32).flatten()

        # Scores rotated, normalised
        all_scores = self.compute_scores()
        scores_rot = (all_scores[perm] / MAX_ECO_SCORE).astype(np.float32)

        # Draw pile
        draw_norm = np.array(
            [len(s.draw_pile) / max(TOTAL_DECK, 1)], dtype=np.float32
        )

        draw_comp = np.zeros((NUM_COLORS, NUM_TYPES), dtype=np.float32)
        for color, typ in s.draw_pile:
            draw_comp[color, typ] += 1
        draw_comp[:, 0] /= max(SINGLES_PER_COLOR, 1)
        draw_comp[:, 1] /= max(DOUBLES_PER_COLOR, 1)

        return RÖkoObs(
            current_player=np.array([0], dtype=np.int32),
            phase=np.array([s.phase], dtype=np.int32),
            hands=hands_flat,
            recycling_side=rec_flat,
            waste_side=waste_flat,
            factory_stacks=factory_slots.flatten(),
            collected=coll.flatten(),
            penalty_pile=pen_norm,
            scores=scores_rot,
            draw_pile_size=draw_norm,
            draw_pile_comp=draw_comp.flatten(),
        )

    @property
    def current_player(self) -> int:
        return self.state.current_player

    @property
    def done(self) -> bool:
        return self.state.done

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    # ── Scoring ──────────────────────────────────────────────────────────

    def compute_scores(self) -> np.ndarray:
        """Compute current/final scores for all players."""
        s = self.state
        n = self._num_players
        scores = np.zeros(n, dtype=np.float32)

        for p in range(n):
            for c in range(NUM_COLORS):
                cards = s.collected[p][c]
                if len(cards) > 1:
                    scores[p] += sum(cards)

        penalty_counts = s.penalty_pile.sum(axis=(1, 2)).astype(np.float32)
        scores -= penalty_counts

        players_with_penalty = int((penalty_counts > 0).sum())
        if players_with_penalty > 0:
            bonus_map = {2: 3, 3: 2, 4: 1, 5: 1}
            bonus = bonus_map.get(n, 1)
            for p in range(n):
                if penalty_counts[p] == 0:
                    scores[p] += bonus

        return scores

    # ── Step helpers ─────────────────────────────────────────────────────

    def _step_play(self, action: int) -> np.ndarray:
        s = self.state
        p = s.current_player
        color, n_s, n_d = decode_play(action)
        rewards = np.zeros(self._num_players, dtype=np.float32)

        s.hands[p, color, 0] -= n_s
        s.hands[p, color, 1] -= n_d
        s.recycling_side[color, 0] += n_s
        s.recycling_side[color, 1] += n_d

        total = int(
            s.recycling_side[color, 0] * CARD_VALUES[0]
            + s.recycling_side[color, 1] * CARD_VALUES[1]
        )

        if total >= MIN_RECYCLE_VALUE:
            if s.factory_stacks[color]:
                fcard = s.factory_stacks[color].pop(0)
                s.collected[p][color].append(fcard)
                rewards[p] += fcard / MAX_ECO_SCORE
                if not s.factory_stacks[color]:
                    s.pending_done = True

            s.discard_counts[color, 0] += s.recycling_side[color, 0]
            s.discard_counts[color, 1] += s.recycling_side[color, 1]
            s.recycling_side[color] = 0
            draw_count = 1
        else:
            draw_count = 1 + total

        s.hands[p] += s.waste_side[color]
        s.waste_side[color] = np.zeros((NUM_COLORS, NUM_TYPES), dtype=np.int32)

        s.played_color = color
        s.pending_draw_count = draw_count

        if int(s.hands[p].sum()) > HAND_LIMIT:
            s.phase = PHASE_DISCARD
        else:
            rewards += self._end_of_turn()

        return rewards

    def _step_discard(self, action: int) -> np.ndarray:
        s = self.state
        p = s.current_player
        rewards = np.zeros(self._num_players, dtype=np.float32)

        color, t = decode_discard(action)
        s.hands[p, color, t] -= 1
        s.penalty_pile[p, color, t] += 1
        rewards[p] -= 1.0 / MAX_ECO_SCORE

        if int(s.hands[p].sum()) <= HAND_LIMIT:
            rewards += self._end_of_turn()

        return rewards

    def _end_of_turn(self) -> np.ndarray:
        s = self.state
        n = self._num_players
        rewards = np.zeros(n, dtype=np.float32)

        self._refill_waste(s.played_color, s.pending_draw_count)

        if s.pending_done:
            s.done = True
            # Terminal: +1 win, -1 loss (per player)
            scores = self.compute_scores()
            best = float(scores.max())
            for p in range(n):
                rewards[p] = 1.0 if scores[p] >= best else -1.0
        else:
            s.current_player = (s.current_player + 1) % n
            s.phase = PHASE_PLAY

        return rewards

    # ── Deck / draw helpers ──────────────────────────────────────────────

    @staticmethod
    def _build_deck(rng: np.random.Generator) -> List[Tuple[int, int]]:
        deck = []
        for c in range(NUM_COLORS):
            deck.extend([(c, 0)] * SINGLES_PER_COLOR)
            deck.extend([(c, 1)] * DOUBLES_PER_COLOR)
        # np.random.Generator.shuffle works on mutable sequences (lists)
        rng.shuffle(deck)
        return deck

    @staticmethod
    def _draw_into(draw_pile: List, target: np.ndarray) -> bool:
        if not draw_pile:
            return False
        c, t = draw_pile.pop()
        target[c, t] += 1
        return True

    def _refill_waste(self, factory_color: int, count: int):
        s = self.state
        for _ in range(count):
            if not s.draw_pile:
                self._reshuffle()
                if not s.draw_pile:
                    s.done = True
                    return
            self._draw_into(s.draw_pile, s.waste_side[factory_color])

    def _reshuffle(self):
        s = self.state
        if s.discard_counts.sum() == 0:
            return
        deck = []
        for c in range(NUM_COLORS):
            for t in range(NUM_TYPES):
                deck.extend([(c, t)] * int(s.discard_counts[c, t]))
        self.rng.shuffle(deck)
        s.draw_pile = deck
        s.discard_counts[:] = 0
