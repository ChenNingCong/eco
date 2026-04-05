"""
Ticket to Ride (US) game engine — implements BaseGameEngine.

Action space (141 total):
  0        : DrawRandomCard
  1-9      : DrawVisibleCard(color 0-8, including WILD=8)
  10       : DrawDestinations
  11-40    : SelectDestination(dest 0-29)
  41       : FinishSelectingDestinations
  42-140   : ClaimRoute(route 0-98)

Turn flow:
  INIT → ClaimRoute → FINISHED
  INIT → DrawCard → DRAWING_CARDS → DrawCard → FINISHED
  INIT → DrawWild(visible) → FINISHED  (wild from visible ends turn)
  INIT → DrawDestinations → SELECTING → Select... → FinishSelecting → FINISHED
  FIRST_ROUND: INIT → DrawDestinations → SELECTING → Select(min 2) → Finish → FINISHED

Game states: FIRST_ROUND → PLAYING → LAST_ROUND → GAME_OVER
"""

import numpy as np
from typing import NamedTuple
from dataclasses import dataclass, field

from abstract import BaseGameEngine

from .map_data import (
    NUM_COLORS, WILD, CARDS_PER_COLOR, TOTAL_CARDS,
    NUM_ROUTES, ROUTE_COLOR, ROUTE_LENGTH, ROUTE_ADJACENT, ROUTE_POINTS_LIST,
    ROUTE_CITY1, ROUTE_CITY2,
    NUM_DESTINATIONS, DEST_CITY1, DEST_CITY2, DEST_POINTS,
    NUM_CITIES, STARTING_TRAINS, STARTING_HAND_SIZE,
    VISIBLE_CARD_SLOTS, DEST_DRAW_SIZE, LAST_ROUND_THRESHOLD,
    MAX_POINTS,
)

# ── Game phases ──────────────────────────────────────────────────────────────

GAME_FIRST_ROUND = 0
GAME_PLAYING = 1
GAME_LAST_ROUND = 2
GAME_OVER = 3

TURN_INIT = 0
TURN_SELECTING_DEST = 1
TURN_DRAWING_CARDS = 2
TURN_FINISHED = 3

# ── Action encoding ──────────────────────────────────────────────────────────

ACT_DRAW_RANDOM = 0
ACT_DRAW_VISIBLE_BASE = 1    # 1..9 for colors 0..8
ACT_DRAW_DEST = 10
ACT_SELECT_DEST_BASE = 11    # 11..40 for dest 0..29
ACT_FINISH_SELECT = 41
ACT_CLAIM_ROUTE_BASE = 42    # 42..140 for route 0..98
NUM_ACTIONS = ACT_CLAIM_ROUTE_BASE + NUM_ROUTES  # 141

# ── Observation ──────────────────────────────────────────────────────────────

class TTRObs(NamedTuple):
    """Observation for Ticket to Ride, rotated so observing player is at index 0."""
    # Embedding indices (int32)
    game_state: np.ndarray       # (1,)  0-3
    turn_state: np.ndarray       # (1,)  0-3

    # Float features (float32), all player-indexed fields rotated (own=index 0)
    hands: np.ndarray            # (num_players * 9,) card counts per color per player, rotated / 14
    player_trains: np.ndarray    # (num_players,) trains per player / 45
    player_points: np.ndarray    # (num_players,) points per player / MAX_POINTS
    player_dest_counts: np.ndarray  # (num_players,) destination counts / NUM_DESTINATIONS
    visible_cards: np.ndarray    # (9,)  visible card counts per color / VISIBLE_CARD_SLOTS
    deck_size: np.ndarray        # (1,)  / TOTAL_CARDS
    route_ownership: np.ndarray  # (num_players * 99,) per-player binary ownership, rotated
    own_dest_status: np.ndarray  # (30,) 0=not owned, 0.5=uncompleted, 1=completed
    avail_dest: np.ndarray       # (30,) binary, currently available for selection
    dest_selected: np.ndarray    # (1,)  num destinations selected this turn / DEST_DRAW_SIZE


def float_dim(num_players: int = 2) -> int:
    """Total float feature dimension."""
    return (
        num_players * 9           # hands (all players, rotated)
        + num_players             # player_trains
        + num_players             # player_points
        + num_players             # player_dest_counts
        + 9                       # visible_cards
        + 1                       # deck_size
        + num_players * NUM_ROUTES  # route_ownership
        + 30                      # own_dest_status
        + 30                      # avail_dest
        + 1                       # dest_selected
    )  # e.g. 2p=280, 3p=390, 5p=610


# ── Game state ───────────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    hand: np.ndarray          # (9,) int32, card counts per color
    trains: int = STARTING_TRAINS
    points: int = 0
    routes: list = field(default_factory=list)           # list of route_ids
    uncompleted_dest: list = field(default_factory=list)  # list of dest_ids
    completed_dest: list = field(default_factory=list)    # list of dest_ids


# ── Adjacency for destination path-finding ───────────────────────────────────

def _build_adjacency():
    """Build city adjacency from route data (static, called once)."""
    adj = [[] for _ in range(NUM_CITIES)]
    for rid in range(NUM_ROUTES):
        c1, c2 = ROUTE_CITY1[rid], ROUTE_CITY2[rid]
        adj[c1].append((c2, rid))
        adj[c2].append((c1, rid))
    return adj

_CITY_ADJ = _build_adjacency()


def _check_dest_connected(dest_id: int, player_routes: set) -> bool:
    """BFS to check if dest cities are connected through player's claimed routes."""
    c1 = DEST_CITY1[dest_id]
    c2 = DEST_CITY2[dest_id]
    if c1 == c2:
        return True
    visited = set()
    queue = [c1]
    visited.add(c1)
    while queue:
        city = queue.pop()
        for neighbor, rid in _CITY_ADJ[city]:
            if rid in player_routes and neighbor not in visited:
                if neighbor == c2:
                    return True
                visited.add(neighbor)
                queue.append(neighbor)
    return False


# ── Engine ───────────────────────────────────────────────────────────────────

class TTREngine(BaseGameEngine[TTRObs]):
    """
    Ticket to Ride (US) game engine for 2 players.
    """

    def __init__(self, rng: np.random.Generator, num_players: int = 2):
        super().__init__(rng)
        assert 2 <= num_players <= 5, "num_players must be 2-5"
        self._num_players = num_players
        self._game_state = GAME_OVER
        self._turn_state = TURN_FINISHED
        self._current_player = 0
        self._players: list[PlayerState] = []
        self._deck = np.zeros(NUM_COLORS, dtype=np.int32)
        self._visible = np.zeros(NUM_COLORS, dtype=np.int32)
        self._route_owner = np.full(NUM_ROUTES, -1, dtype=np.int32)  # -1=unclaimed
        self._dest_available = []  # dest_ids available for selection
        self._dest_selected_count = 0  # how many selected this turn
        self._last_turn_count = 10000
        self._turn_count = 0
        self._done = True

    def _reset(self) -> None:
        rng = self.rng
        n = self._num_players

        # Deck
        self._deck = np.array(CARDS_PER_COLOR, dtype=np.int32)

        # Players
        self._players = []
        for _ in range(n):
            hand = np.zeros(NUM_COLORS, dtype=np.int32)
            for _ in range(STARTING_HAND_SIZE):
                card = self._draw_card_from_deck()
                if card >= 0:
                    hand[card] += 1
            self._players.append(PlayerState(hand=hand))

        # Visible cards
        self._visible = np.zeros(NUM_COLORS, dtype=np.int32)
        for _ in range(VISIBLE_CARD_SLOTS):
            card = self._draw_card_from_deck()
            if card >= 0:
                self._visible[card] += 1

        # Routes
        self._route_owner = np.full(NUM_ROUTES, -1, dtype=np.int32)

        # Game state — start in FIRST_ROUND, players must draw destinations
        self._game_state = GAME_FIRST_ROUND
        self._turn_state = TURN_INIT
        self._current_player = 0
        self._dest_available = []
        self._dest_selected_count = 0
        self._last_turn_count = 10000
        self._turn_count = 0
        self._done = False

        # Destinations — all unclaimed at start
        self._dest_claimed = np.zeros(NUM_DESTINATIONS, dtype=np.int32)  # 0=unclaimed

    def step(self, action: int) -> tuple[float, ...]:
        assert not self._done, "Game over"
        assert self.legal_actions()[action], f"Illegal action {action}"

        rewards = [0.0] * self._num_players
        p = self._current_player
        player = self._players[p]

        if action == ACT_DRAW_RANDOM:
            self._do_draw_random(player)
        elif ACT_DRAW_VISIBLE_BASE <= action <= ACT_DRAW_VISIBLE_BASE + 8:
            color = action - ACT_DRAW_VISIBLE_BASE
            self._do_draw_visible(player, color)
        elif action == ACT_DRAW_DEST:
            self._do_draw_destinations()
        elif ACT_SELECT_DEST_BASE <= action < ACT_SELECT_DEST_BASE + NUM_DESTINATIONS:
            dest_id = action - ACT_SELECT_DEST_BASE
            self._do_select_destination(player, dest_id)
        elif action == ACT_FINISH_SELECT:
            self._do_finish_selecting()
        elif ACT_CLAIM_ROUTE_BASE <= action < ACT_CLAIM_ROUTE_BASE + NUM_ROUTES:
            route_id = action - ACT_CLAIM_ROUTE_BASE
            self._do_claim_route(player, p, route_id)
        else:
            raise ValueError(f"Unknown action {action}")

        # Check if turn is finished
        if self._turn_state == TURN_FINISHED:
            self._end_turn()

        # Terminal: +1 win, -1 loss
        if self._done:
            scores = self.compute_scores()
            best = float(scores.max())
            for i in range(self._num_players):
                rewards[i] = 1.0 if scores[i] >= best else -1.0

        return tuple(rewards)

    def legal_actions(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        p = self._current_player
        player = self._players[p]
        gs = self._game_state
        ts = self._turn_state

        if ts == TURN_INIT:
            # Draw destinations (always legal if any unclaimed)
            if np.any(self._dest_claimed == 0):
                mask[ACT_DRAW_DEST] = True

            if gs in (GAME_PLAYING, GAME_LAST_ROUND):
                # Draw random card
                if self._deck.sum() > 0:
                    mask[ACT_DRAW_RANDOM] = True

                # Draw visible card (any color including wild)
                for c in range(NUM_COLORS):
                    if self._visible[c] > 0:
                        mask[ACT_DRAW_VISIBLE_BASE + c] = True

                # Claim routes
                player_route_set = set(player.routes)
                for rid in range(NUM_ROUTES):
                    if self._route_owner[rid] >= 0:
                        continue
                    if player.trains < ROUTE_LENGTH[rid]:
                        continue
                    # Double route rules:
                    # - Same player can never claim both routes of a pair
                    # - With 2-3 players, only one route of a pair can be claimed total
                    adj = ROUTE_ADJACENT[rid]
                    if adj >= 0:
                        if adj in player_route_set:
                            continue
                        if self._num_players <= 3 and self._route_owner[adj] >= 0:
                            continue
                    # Payment check
                    if self._can_pay(player, rid):
                        mask[ACT_CLAIM_ROUTE_BASE + rid] = True

        elif ts == TURN_SELECTING_DEST:
            # Select any available destination
            for did in self._dest_available:
                mask[ACT_SELECT_DEST_BASE + did] = True
            # Finish selecting (need min 2 in first round, min 1 otherwise)
            min_sel = 2 if gs == GAME_FIRST_ROUND else 1
            if self._dest_selected_count >= min_sel:
                mask[ACT_FINISH_SELECT] = True

        elif ts == TURN_DRAWING_CARDS:
            # Second card draw — can draw random or visible non-wild
            if self._deck.sum() > 0:
                mask[ACT_DRAW_RANDOM] = True
            for c in range(NUM_COLORS):
                if c == WILD:
                    continue  # can't draw visible wild as second draw
                if self._visible[c] > 0:
                    mask[ACT_DRAW_VISIBLE_BASE + c] = True

        return mask

    def encode(self, player_id: int) -> TTRObs:
        p = player_id
        n = self._num_players
        # Rotation permutation: observing player at index 0
        perm = [(p + i) % n for i in range(n)]

        # Hands: all players visible, rotated (num_players * 9,)
        hands = np.concatenate([
            self._players[pi].hand.astype(np.float32) / 14.0 for pi in perm
        ])

        # Per-player stats, rotated
        player_trains = np.array([self._players[pi].trains / STARTING_TRAINS for pi in perm], dtype=np.float32)
        player_points = np.array([self._players[pi].points / MAX_POINTS for pi in perm], dtype=np.float32)
        player_dest_counts = np.array([
            (len(self._players[pi].uncompleted_dest) + len(self._players[pi].completed_dest)) / NUM_DESTINATIONS
            for pi in perm
        ], dtype=np.float32)

        # Route ownership: per-player binary masks (num_players * 99,), rotated
        route_ownership = np.zeros((n, NUM_ROUTES), dtype=np.float32)
        for rid in range(NUM_ROUTES):
            owner = self._route_owner[rid]
            if owner >= 0:
                # Map absolute owner to rotated index
                rotated_idx = (owner - p) % n
                route_ownership[rotated_idx, rid] = 1.0

        # Destination status for observing player only (private info)
        player = self._players[p]
        own_dest_status = np.zeros(NUM_DESTINATIONS, dtype=np.float32)
        for did in player.uncompleted_dest:
            own_dest_status[did] = 0.5
        for did in player.completed_dest:
            own_dest_status[did] = 1.0

        # Available destinations for selection
        avail_dest = np.zeros(NUM_DESTINATIONS, dtype=np.float32)
        for did in self._dest_available:
            avail_dest[did] = 1.0

        return TTRObs(
            game_state=np.array([self._game_state], dtype=np.int32),
            turn_state=np.array([self._turn_state], dtype=np.int32),
            hands=hands,
            player_trains=player_trains,
            player_points=player_points,
            player_dest_counts=player_dest_counts,
            visible_cards=self._visible.astype(np.float32) / VISIBLE_CARD_SLOTS,
            deck_size=np.array([self._deck.sum() / TOTAL_CARDS], dtype=np.float32),
            route_ownership=route_ownership.flatten(),
            own_dest_status=own_dest_status,
            avail_dest=avail_dest,
            dest_selected=np.array([self._dest_selected_count / DEST_DRAW_SIZE], dtype=np.float32),
        )

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def done(self) -> bool:
        return self._done

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    # ── Action implementations ───────────────────────────────────────────

    def _do_draw_random(self, player: PlayerState):
        card = self._draw_card_from_deck()
        if card >= 0:
            player.hand[card] += 1
        if self._turn_state == TURN_INIT:
            self._turn_state = TURN_DRAWING_CARDS
        elif self._turn_state == TURN_DRAWING_CARDS:
            self._turn_state = TURN_FINISHED

    def _do_draw_visible(self, player: PlayerState, color: int):
        assert self._visible[color] > 0
        self._visible[color] -= 1
        player.hand[color] += 1
        self._replenish_visible()

        if color == WILD:
            # Drawing wild from visible always ends turn
            self._turn_state = TURN_FINISHED
        elif self._turn_state == TURN_INIT:
            self._turn_state = TURN_DRAWING_CARDS
        elif self._turn_state == TURN_DRAWING_CARDS:
            self._turn_state = TURN_FINISHED

    def _do_draw_destinations(self):
        unclaimed = [i for i in range(NUM_DESTINATIONS) if self._dest_claimed[i] == 0]
        sample_size = min(DEST_DRAW_SIZE, len(unclaimed))
        chosen = self.rng.choice(unclaimed, size=sample_size, replace=False).tolist()
        self._dest_available = chosen
        self._dest_selected_count = 0
        self._turn_state = TURN_SELECTING_DEST

    def _do_select_destination(self, player: PlayerState, dest_id: int):
        assert dest_id in self._dest_available
        self._dest_available.remove(dest_id)
        self._dest_claimed[dest_id] = 1
        player.uncompleted_dest.append(dest_id)
        self._dest_selected_count += 1
        # Check if destination is already completed
        player_route_set = set(player.routes)
        if _check_dest_connected(dest_id, player_route_set):
            player.uncompleted_dest.remove(dest_id)
            player.completed_dest.append(dest_id)

    def _do_finish_selecting(self):
        self._dest_available = []
        self._turn_state = TURN_FINISHED

    def _do_claim_route(self, player: PlayerState, player_idx: int, route_id: int) -> int:
        length = ROUTE_LENGTH[route_id]
        color = ROUTE_COLOR[route_id]
        points = ROUTE_POINTS_LIST[route_id]

        # Find best payment (maximize colored cards, minimize wilds)
        payment = self._best_payment(player, color, length)
        assert payment is not None

        # Apply payment
        for c, count in payment:
            player.hand[c] -= count
            self._deck[c] += count  # cards go back to deck

        self._route_owner[route_id] = player_idx
        player.routes.append(route_id)
        player.trains -= length
        player.points += points

        self._replenish_visible()

        # Check destination completions
        player_route_set = set(player.routes)
        newly_completed = []
        for did in player.uncompleted_dest:
            if _check_dest_connected(did, player_route_set):
                newly_completed.append(did)
        for did in newly_completed:
            player.uncompleted_dest.remove(did)
            player.completed_dest.append(did)

        self._turn_state = TURN_FINISHED
        return points

    # ── Turn/game state transitions ──────────────────────────────────────

    def _end_turn(self):
        """Called when turn_state becomes FINISHED."""
        # State transitions
        if self._game_state == GAME_FIRST_ROUND:
            if self._current_player == self._num_players - 1:
                self._game_state = GAME_PLAYING
        elif self._game_state == GAME_PLAYING:
            if any(p.trains < LAST_ROUND_THRESHOLD for p in self._players):
                self._game_state = GAME_LAST_ROUND
                self._last_turn_count = self._turn_count + self._num_players
        elif self._game_state == GAME_LAST_ROUND:
            if self._turn_count >= self._last_turn_count:
                self._game_state = GAME_OVER
                self._calculate_final_scores()
                self._done = True
                return

        # Advance to next player
        self._turn_count += 1
        self._current_player = (self._current_player + 1) % self._num_players
        self._turn_state = TURN_INIT
        self._dest_selected_count = 0

    def _calculate_final_scores(self):
        """Add/subtract destination points at game end."""
        for player in self._players:
            for did in player.completed_dest:
                player.points += DEST_POINTS[did]
            for did in player.uncompleted_dest:
                player.points -= DEST_POINTS[did]

    # ── Scoring (for terminal rewards) ───────────────────────────────────

    def compute_scores(self) -> np.ndarray:
        return np.array([p.points for p in self._players], dtype=np.float32)

    def game_metrics(self, player_idx: int) -> dict:
        """Return informative metrics for a specific player (call after game ends)."""
        p = self._players[player_idx]
        route_lengths = [ROUTE_LENGTH[rid] for rid in p.routes]
        return {
            "routes_claimed": len(p.routes),
            "trains_remaining": p.trains,
            "dest_completed": len(p.completed_dest),
            "dest_failed": len(p.uncompleted_dest),
            "dest_total": len(p.completed_dest) + len(p.uncompleted_dest),
            "route_points": sum(ROUTE_POINTS_LIST[rid] for rid in p.routes),
            "dest_points": sum(DEST_POINTS[did] for did in p.completed_dest),
            "dest_penalty": sum(DEST_POINTS[did] for did in p.uncompleted_dest),
            "avg_route_length": np.mean(route_lengths) if route_lengths else 0.0,
            "max_route_length": max(route_lengths) if route_lengths else 0,
            "total_points": p.points,
        }

    # ── Card/payment helpers ─────────────────────────────────────────────

    def _draw_card_from_deck(self) -> int:
        """Draw a random card from deck. Returns color index, or -1 if empty."""
        total = self._deck.sum()
        if total == 0:
            return -1
        probs = self._deck.astype(np.float64) / total
        color = int(self.rng.choice(NUM_COLORS, p=probs))
        self._deck[color] -= 1
        return color

    def _replenish_visible(self):
        """Refill visible cards to 5 from deck."""
        while self._visible.sum() < VISIBLE_CARD_SLOTS and self._deck.sum() > 0:
            card = self._draw_card_from_deck()
            if card >= 0:
                self._visible[card] += 1

    def _can_pay(self, player: PlayerState, route_id: int) -> bool:
        return self._best_payment(player, ROUTE_COLOR[route_id], ROUTE_LENGTH[route_id]) is not None

    def _best_payment(self, player: PlayerState, color: int, length: int):
        """
        Find best payment for a route. Returns list of (color, count) or None.
        For colored routes: use colored cards + wilds (prefer colored).
        For WILD (gray) routes: try each color, pick the one using fewest wilds.
        """
        if color != WILD:
            return self._try_payment(player, color, length)
        else:
            # Gray route: try every non-wild color
            best = None
            best_wilds = length + 1
            for c in range(NUM_COLORS - 1):  # skip WILD itself
                pay = self._try_payment(player, c, length)
                if pay is not None:
                    wilds_used = sum(cnt for col, cnt in pay if col == WILD)
                    if wilds_used < best_wilds:
                        best = pay
                        best_wilds = wilds_used
            # Also try all-wilds
            if player.hand[WILD] >= length:
                if best is None or length < best_wilds:
                    best = [(WILD, length)]
            return best

    def _try_payment(self, player: PlayerState, color: int, length: int):
        """Try to pay with colored cards + wilds. Returns [(color, n), (WILD, m)] or None."""
        have_color = int(player.hand[color])
        have_wild = int(player.hand[WILD])
        use_color = min(have_color, length)
        use_wild = length - use_color
        if use_wild > have_wild:
            return None
        result = []
        if use_color > 0:
            result.append((color, use_color))
        if use_wild > 0:
            result.append((WILD, use_wild))
        return result
