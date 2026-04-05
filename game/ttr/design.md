# Ticket to Ride (USA) — Game Engine & Agent Design

## Game Overview

Ticket to Ride (US map, the original base game) for 2-5 players. Players collect train cards, claim routes between cities on a map of North America, and complete destination tickets for bonus points.

## Map

| Quantity | Value |
|----------|-------|
| Cities | 36 (US/Canada) |
| Routes | 99 (connections between cities) |
| Double routes | 22 pairs (same city pair, different colors) |
| Destinations | 30 (city-to-city tickets) |
| Train colors | 8 + WILD(gray) = 9 |
| Total train cards | 110 (12 each color, 14 wild) |
| Starting trains per player | 45 |
| Starting cards per player | 4 |

## Turn Flow (State Machine)

```
FIRST_ROUND:
  INIT → DrawDestinations → SELECTING_DEST → Select(min 2) → FinishSelecting → FINISHED
  (each player does this once, then → GAME_PLAYING)

PLAYING / LAST_ROUND:
  INIT → ClaimRoute → FINISHED                          (claim a route, pay cards)
  INIT → DrawCard → DRAWING_CARDS → DrawCard → FINISHED (draw 2 cards total)
  INIT → DrawWild(visible) → FINISHED                   (wild from visible = 1 card only)
  INIT → DrawDestinations → SELECTING_DEST → Select(min 1) → FinishSelecting → FINISHED

LAST_ROUND triggers when any player has < 3 trains.
GAME_OVER after all players have one more full turn.
```

### Select Destination

When you choose "Draw Destinations", 3 random unclaimed destination tickets are revealed. You must keep at least 2 (first round) or 1 (later rounds). Each destination is a city pair with a point value — you score those points at game end if you connect the cities through your claimed routes, but LOSE those points if you fail.

### Claim Route

Routes are colored connections between two cities (length 1-6). To claim one, you pay train cards matching the route's color and length. **Gray routes** accept any single color. Payment is auto-selected (maximize colored cards, minimize wilds). You immediately score points based on route length.

## Action Space (141 total)

| Action ID | Description |
|-----------|-------------|
| 0 | DrawRandomCard (from deck) |
| 1-9 | DrawVisibleCard(color 0-8, including WILD=8) |
| 10 | DrawDestinations (draw 3 from unclaimed) |
| 11-40 | SelectDestination(dest 0-29) |
| 41 | FinishSelectingDestinations |
| 42-140 | ClaimRoute(route 0-98) |

**Wild card rules:**
- Drawing visible wild in INIT → turn ends (only 1 card)
- Drawing visible wild in DRAWING_CARDS → NOT legal (must draw non-wild or from deck)
- Drawing from deck → always legal, no restrictions

**Route claiming:**
- Payment auto-selected: maximize colored cards, minimize wilds
- Gray (WILD) routes: engine tries all 8 colors + wilds, picks best payment
- Double routes: same player can never claim both. With 2-3 players, only one of the pair can be claimed total. With 4-5 players, both can be claimed by different players.

## Observation Space (N = num_players)

Total: 2 embedding + variable float dims (293 for 2p, 400 for 3p, 614 for 5p)

### Embedding indices (int32)
| Field | Shape | Values |
|-------|-------|--------|
| game_state | (1,) | 0=FIRST_ROUND, 1=PLAYING, 2=LAST_ROUND, 3=GAME_OVER |
| turn_state | (1,) | 0=INIT, 1=SELECTING_DEST, 2=DRAWING_CARDS, 3=FINISHED |

### Float features (float32, normalized, player-indexed fields rotated so self=index 0)
| Field | Shape | Description | Normalization |
|-------|-------|-------------|---------------|
| hands | (N*9,) | All players' card counts per color (public) | /14 |
| player_trains | (N,) | Trains remaining per player | /45 |
| player_points | (N,) | Points per player | /300 |
| player_dest_counts | (N,) | Destination count per player (count only, not which) | /30 |
| visible_cards | (9,) | Face-up card counts per color | /5 |
| deck_size | (1,) | Cards remaining in deck | /110 |
| route_ownership | (N*99,) | Per-player binary route ownership, rotated | binary |
| own_dest_status | (30,) | Own destination status (private) | 0/0.5/1 |
| avail_dest | (30,) | Available for selection this turn | binary |
| dest_selected | (1,) | Destinations selected this turn | /3 |

**Design principles:**
- All player-indexed data rotated: current player always at index 0
- Hands are **public** (all players' card breakdowns visible) — destinations stay private
- Route ownership uses per-player binary channels (N blocks of 99), not signed encoding — scales naturally with player count
- Normalization to ~[0,1] for stable training

## Agent Architecture

```
TTRAgent (~3M params for 2p):
  Embeddings: game_state(4→32), turn_state(4→32)
  Encoder: 3-layer FF (float_dim→512→512→512) + LayerNorm + ReLU
  fusion_in = 512 + 64 = 576
  LSTM: parallel path (576→256)
  concat(fusion, lstm_out) = 576 + 256 = 832
  Actor trunk: 2-layer (832→512→512) + LayerNorm + ReLU → head(141)
  Critic trunk: 2-layer (832→512→512) + LayerNorm + ReLU → head(1)
```

Compared to R-Öko agent (720K params):
- 3-layer encoder (vs 2) — more complex observation
- Hidden width 512 (vs 256) — larger action space needs more capacity
- LSTM hidden 256 (vs 128) — longer games, more temporal reasoning needed

## Scoring

**Route points** (immediate when claimed):

| Length | Points |
|--------|--------|
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 4 | 7 |
| 5 | 10 |
| 6 | 15 |

**Destination points** (end of game):
- Completed (cities connected through claimed routes): +points (4 to 22)
- Uncompleted: -points (penalty)

## Multi-player Support

| Players | Double route rule | Observation float dims |
|---------|-------------------|----------------------|
| 2 | Only 1 of pair claimable | 293 |
| 3 | Only 1 of pair claimable | 400 |
| 4 | Both claimable (different players) | 507 |
| 5 | Both claimable (different players) | 614 |

## Key Differences from R-Öko

| Aspect | R-Öko | Ticket to Ride |
|--------|-------|----------------|
| Players | 2-5 | 2-5 |
| Action space | 108 | 141 |
| Obs float dims (2p) | ~150 | 293 |
| Game length | 16-42 steps | 50-200+ steps |
| Turn structure | Single action | Multi-step (draw 2 cards, select destinations) |
| Info visibility | Opponent hands hidden | Hands public, destinations private |
| Reward structure | Terminal (+1/-1) | Terminal (+1/-1 based on final score) |

## Benchmark Metrics

All metrics are averaged over 100 games of the trained agent vs a random opponent.

### Win rate
| Metric | Description | Range |
|--------|-------------|-------|
| `benchmark/win_rate_vs_random` | Fraction of games won (highest score) | 0.0 – 1.0 |
| `benchmark/mean_score` | Agent's average final score | ~ -200 to 300 |
| `benchmark/std_score` | Std dev of agent's final score | ≥ 0 |
| `benchmark/mean_reward` | Agent's average terminal reward | -1.0 to 1.0 |

### Game-level metrics (from `game_metrics`)
| Metric | Description | Range | Good agent |
|--------|-------------|-------|------------|
| `benchmark/routes_claimed` | Routes claimed per game | 0 – ~20 | 10–20 |
| `benchmark/trains_remaining` | Trains left at game end | 0 – 45 | 0–5 (uses trains efficiently) |
| `benchmark/dest_completed` | Destination tickets completed | 0 – 30 | High |
| `benchmark/dest_failed` | Destination tickets failed (penalty) | 0 – 30 | 0 |
| `benchmark/dest_total` | Total destinations held | 0 – 30 | 3–8 |
| `benchmark/route_points` | Points from route claims only | 0 – ~150 | 50–100 |
| `benchmark/dest_points` | Points from completed destinations | 0 – ~200 | 30–80 |
| `benchmark/dest_penalty` | Points lost from failed destinations | 0 – ~200 | 0 |
| `benchmark/avg_route_length` | Mean length of claimed routes | 1.0 – 6.0 | 2.5–4.0 |
| `benchmark/max_route_length` | Longest route claimed | 0 – 6 | 5–6 |
| `benchmark/total_points` | Final score (route + dest - penalty) | ~ -200 to 300 | 80–200 |

### Scoring breakdown
`total_points = route_points + dest_points - dest_penalty`

- **route_points**: Sum of points from all claimed routes (length → points: 1→1, 2→2, 3→4, 4→7, 5→10, 6→15)
- **dest_points**: Sum of point values of completed destination tickets (4–22 pts each)
- **dest_penalty**: Sum of point values of uncompleted destination tickets (subtracted from score)
