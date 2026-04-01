"""
Hearts Card Game Environment for Reinforcement Learning.

Core game logic without observation encoding.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field


# ── Card constants ──────────────────────────────────────────────────────────
NUM_CARDS = 52
NUM_PLAYERS = 4
NUM_ROUNDS = 13
NUM_SUITS = 4
CARDS_PER_SUIT = 13

# Suits: 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades
CLUBS, DIAMONDS, HEARTS, SPADES = 0, 1, 2, 3

# Card encoding: card_id = suit * 13 + rank  (rank 0=2, ..., 12=Ace)
def card_suit(card_id: int) -> int:
    return card_id // CARDS_PER_SUIT

def card_rank(card_id: int) -> int:
    return card_id % CARDS_PER_SUIT

def make_card(suit: int, rank: int) -> int:
    return suit * CARDS_PER_SUIT + rank

# Special cards
TWO_OF_CLUBS   = make_card(CLUBS, 0)       # 0
QUEEN_OF_SPADES = make_card(SPADES, 10)    # 49
JACK_OF_DIAMONDS = make_card(DIAMONDS, 9)  # 22

def card_points(card_id: int) -> int:
    """Return penalty points for a card."""
    if card_suit(card_id) == HEARTS:
        return 1
    if card_id == QUEEN_OF_SPADES:
        return 13
    return 0

MAX_SCORE = 26  # 13 hearts + queen of spades
FLAT_DIM  = 4 + 1 + 52 + 52 + 4 + 1  # 114


# ── Round history entry ─────────────────────────────────────────────────────
@dataclass
class RoundRecord:
    """Records one completed trick."""
    leading_player: int                  # 0-3
    cards: np.ndarray                    # shape (4,) card ids, -1 = not played
    players: np.ndarray                  # shape (4,) player ids, -1 = not played
    winner: int = -1


# ── Game state ──────────────────────────────────────────────────────────────
@dataclass
class HeartsState:
    hands: np.ndarray                    # (4, 52) bool bitmap
    scores: np.ndarray                   # (4,) int
    round_num: int                       # 0-12
    current_player: int
    leading_player: int
    current_trick_cards: np.ndarray      # (4,) -1 or card_id (indexed by play order)
    current_trick_players: np.ndarray    # (4,) -1 or player_id
    current_trick_count: int             # how many cards played in trick so far
    hearts_broken: bool
    played_cards: np.ndarray             # (52,) bool bitmap of all played cards
    history: list = field(default_factory=list)  # list of RoundRecord
    done: bool = False
    points_this_trick: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int32))

    def leading_suit(self) -> int:
        """Return suit of the leading card, or -1 if trick not started."""
        if self.current_trick_count == 0:
            return -1
        first_card = self.current_trick_cards[0]
        return card_suit(first_card)


class HeartsEnv:
    """
    Single-instance Hearts card game environment.

    Observations are raw dicts (see `observe()`).
    Use `HeartsEnvWrapper` for neural-network-ready encoded observations.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.state: Optional[HeartsState] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self) -> "HeartsState":
        deck = np.arange(NUM_CARDS)
        self.rng.shuffle(deck)
        hands = np.zeros((NUM_PLAYERS, NUM_CARDS), dtype=bool)
        for p in range(NUM_PLAYERS):
            hands[p, deck[p * NUM_ROUNDS:(p + 1) * NUM_ROUNDS]] = True

        # Find who has 2 of clubs
        starting_player = int(np.argmax(hands[:, TWO_OF_CLUBS]))

        self.state = HeartsState(
            hands=hands,
            scores=np.zeros(NUM_PLAYERS, dtype=np.int32),
            round_num=0,
            current_player=starting_player,
            leading_player=starting_player,
            current_trick_cards=np.full(NUM_PLAYERS, -1, dtype=np.int32),
            current_trick_players=np.full(NUM_PLAYERS, -1, dtype=np.int32),
            current_trick_count=0,
            hearts_broken=False,
            played_cards=np.zeros(NUM_CARDS, dtype=bool),
            history=[],
        )
        return self.state

    def legal_actions(self, state: Optional[HeartsState] = None) -> np.ndarray:
        """Return boolean mask (52,) of legal card ids for current player."""
        s = state or self.state
        p = s.current_player
        hand = s.hands[p].copy()  # cards in hand

        if not hand.any():
            return hand

        # First trick: must lead/follow with 2 of clubs if holding it
        if s.round_num == 0 and s.current_trick_count == 0:
            if hand[TWO_OF_CLUBS]:
                mask = np.zeros(NUM_CARDS, dtype=bool)
                mask[TWO_OF_CLUBS] = True
                return mask

        leading_suit = s.leading_suit()

        if leading_suit == -1:
            # Leading the trick
            if not s.hearts_broken:
                # Cannot lead hearts unless only hearts left
                non_hearts = hand.copy()
                non_hearts[HEARTS * CARDS_PER_SUIT:(HEARTS + 1) * CARDS_PER_SUIT] = False
                if non_hearts.any():
                    return non_hearts
            return hand
        else:
            # Must follow suit if possible
            suit_cards = np.zeros(NUM_CARDS, dtype=bool)
            suit_cards[leading_suit * CARDS_PER_SUIT:(leading_suit + 1) * CARDS_PER_SUIT] = True
            follow = hand & suit_cards
            if follow.any():
                # First trick: cannot play hearts or queen of spades
                if s.round_num == 0:
                    no_penalty = follow.copy()
                    no_penalty[HEARTS * CARDS_PER_SUIT:(HEARTS + 1) * CARDS_PER_SUIT] = False
                    no_penalty[QUEEN_OF_SPADES] = False
                    if no_penalty.any():
                        return no_penalty
                return follow
            else:
                # Cannot follow suit; can play anything except…
                # First trick: no hearts/queen of spades if possible
                if s.round_num == 0:
                    no_penalty = hand.copy()
                    no_penalty[HEARTS * CARDS_PER_SUIT:(HEARTS + 1) * CARDS_PER_SUIT] = False
                    no_penalty[QUEEN_OF_SPADES] = False
                    if no_penalty.any():
                        return no_penalty
                return hand

    def step(self, action: int):
        """
        Play `action` (card_id) for the current player.
        Returns (state, rewards, done, info).
        rewards is a (4,) array of per-player reward signals for this step.
        """
        s = self.state
        assert not s.done, "Game is over; call reset()."
        mask = self.legal_actions(s)
        assert mask[action], f"Illegal action {action} for player {s.current_player}"

        p = s.current_player
        pos = s.current_trick_count

        # Play the card
        s.hands[p, action] = False
        s.played_cards[action] = True
        s.current_trick_cards[pos] = action
        s.current_trick_players[pos] = p
        s.current_trick_count += 1

        # Track hearts broken
        if card_suit(action) == HEARTS:
            s.hearts_broken = True

        rewards = np.zeros(NUM_PLAYERS, dtype=np.float32)

        if s.current_trick_count == NUM_PLAYERS:
            # Resolve trick
            winner, trick_points = self._resolve_trick(s)
            s.scores[winner] += trick_points

            # Check shoot-the-moon
            rewards = self._compute_rewards(s, winner, trick_points)

            # Save history
            record = RoundRecord(
                leading_player=s.leading_player,
                cards=s.current_trick_cards.copy(),
                players=s.current_trick_players.copy(),
                winner=winner,
            )
            s.history.append(record)

            # Advance round
            s.round_num += 1
            s.current_trick_cards = np.full(NUM_PLAYERS, -1, dtype=np.int32)
            s.current_trick_players = np.full(NUM_PLAYERS, -1, dtype=np.int32)
            s.current_trick_count = 0
            s.leading_player = winner
            s.current_player = winner

            if s.round_num == NUM_ROUNDS:
                s.done = True
                # Final shoot-the-moon check already handled per-trick; issue terminal rewards
                rewards = self._terminal_rewards(s)
        else:
            # Advance to next player
            s.current_player = (p + 1) % NUM_PLAYERS

        return s, rewards, s.done, {}

    def observe(self) -> dict:
        """Return raw observation dict for current player."""
        s = self.state
        return {
            "hands": s.hands.copy(),
            "scores": s.scores.copy(),
            "round_num": s.round_num,
            "current_player": s.current_player,
            "leading_player": s.leading_player,
            "current_trick_cards": s.current_trick_cards.copy(),
            "current_trick_players": s.current_trick_players.copy(),
            "current_trick_count": s.current_trick_count,
            "hearts_broken": s.hearts_broken,
            "played_cards": s.played_cards.copy(),
            "history": list(s.history),
            "legal_actions": self.legal_actions(),
        }

    # ── Internal helpers ────────────────────────────────────────────────────

    def _resolve_trick(self, s: HeartsState):
        """Determine trick winner and total points."""
        leading_suit = card_suit(s.current_trick_cards[0])
        best_pos = 0
        best_rank = card_rank(s.current_trick_cards[0])

        for pos in range(1, NUM_PLAYERS):
            card = s.current_trick_cards[pos]
            if card_suit(card) == leading_suit:
                rank = card_rank(card)
                if rank > best_rank:
                    best_rank = rank
                    best_pos = pos

        winner = s.current_trick_players[best_pos]
        trick_points = sum(card_points(c) for c in s.current_trick_cards)
        return winner, trick_points

    def _compute_rewards(self, s: HeartsState, trick_winner: int, trick_points: int) -> np.ndarray:
        """Simple intermediate rewards: negative for points taken."""
        rewards = np.zeros(NUM_PLAYERS, dtype=np.float32)
        if trick_points > 0:
            rewards[trick_winner] -= trick_points / MAX_SCORE
        return rewards

    def _terminal_rewards(self, s: HeartsState) -> np.ndarray:
        """Compute final game rewards. Handles shoot-the-moon."""
        scores = s.scores.copy()

        # Shoot the moon: one player got all 26 points
        shooter = np.where(scores == MAX_SCORE)[0]
        if len(shooter) == 1:
            rewards = np.ones(NUM_PLAYERS, dtype=np.float32)
            rewards[shooter[0]] = -1.0
        else:
            # Reward is negative normalized score
            rewards = -(scores.astype(np.float32) / MAX_SCORE)
        return rewards