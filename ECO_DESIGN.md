# R-öko RL — Design & Usage

## Overview

This project adapts a self-play PPO agent from Hearts to **R-öko**, a recycling-themed
card game. The existing training infrastructure (vectorised environments, PPO loop,
Flask server) is reused with minimal changes; only the game engine, observation encoder,
and agent architecture are new.

---

## File Map

```
eco_env.py          Game engine — pure game logic, no RL
eco_obs_encoder.py  Observation encoding + single-player RL wrapper
eco_vec_env.py      Vectorised environment (N parallel games)
eco_ppo.py          PPO training + EcoAgent neural network
eco_tests.py        Test suite (49 tests)
server.py           Flask server — Hearts + R-öko on the same port
static/eco_index.html  Browser UI for R-öko
eco_rule.md         Game rules + state/action space notes
```

---

## Game Design

### Deck (88 cards)

| Type   | Value | Per colour | Total |
|--------|-------|-----------|-------|
| Single |   1   |    19     |  76   |
| Double |   2   |    3      |  12   |

4 colours: **Glass, Paper, Plastic, Tin**

### Factory stacks (per colour)

| Players | Stack (top → bottom)       |
|---------|----------------------------|
| 2       | 0, 1, 2, −2, 4, 5          |
| 3–4     | 0, 1, 2, 3, −2, 4, 5       |
| 5       | 0, 1, 2, 3, 3, −2, 4, 5   |

Game ends when any colour's stack is exhausted (current player completes their turn).

### Turn structure

```
Phase 0 – PLAY
  Player plays ≥1 recycling cards of one colour on that factory's recycling side.
  • If recycling-side total value ≥ 4 → take top factory card; clear recycling side.
  • Player picks up ALL waste cards from that factory.
  • If hand > 5 → enter Phase 1 (discard), else proceed to refill.

Phase 1 – DISCARD  (repeated until hand ≤ 5)
  Player discards one card (it becomes a public face-down penalty card).

After hand ≤ 5:
  Draw (1 + recycling-side value) new cards into that factory's waste side.
  Advance to next player.
```

### Scoring

1. For each colour a player has **> 1** factory card: add sum of all card values.
2. Subtract **1 point per penalty card**.
3. If any player has penalty cards, players with **zero** penalty cards gain a bonus:
   +3 (2-player), +2 (3-player), +1 (4–5 player).

---

## State Space (perfect information)

All values are observable by both players.

| Component | Shape | Notes |
|-----------|-------|-------|
| `hands` | `(P, 4, 2)` int | Singles/doubles per colour per player |
| `recycling_side` | `(4, 2)` int | Cards played on each factory's recycling side |
| `waste_side` | `(4, 4, 2)` int | Cards on each factory's waste side (any colour) |
| `factory_stacks` | `4 × list` | Remaining stack per colour |
| `collected` | `P × 4 × list` | Factory cards won per player per colour |
| `penalty_pile` | `(P, 4, 2)` int | Public penalty pile composition per player |
| `draw_pile` | `list[(c,t)]` | Remaining draw deck |
| `current_player` | int | |
| `phase` | 0 or 1 | Play or Discard |

*P = num_players (default 2)*

---

## Action Space (108 actions, phase-masked)

### Play actions (0–99)

```
action_id = color × 25 + n_singles × 5 + n_doubles
```

- `color`     ∈ {0 Glass, 1 Paper, 2 Plastic, 3 Tin}
- `n_singles` ∈ {0, 1, 2, 3, 4}
- `n_doubles` ∈ {0, 1, 2, 3, 4}
- At least one card must be played (`n_singles + n_doubles ≥ 1`)

### Discard actions (100–107)

```
action_id = 100 + color × 2 + card_type
```

- `card_type` ∈ {0=single, 1=double}

The `phase` token in the observation tells the agent which 100-action or 8-action
subset is currently legal.

---

## Neural Network — EcoAgent

```
Input:
  current_player  (1,)   int32  → Embedding(P+1, 32)
  phase           (1,)   int32  → Embedding(2,   32)
  float features  (85,)  float32 (2-player):
    hands            16   (2×4×2, normalised)
    recycling_side    8   (4×2)
    waste_side       32   (4×4×2)
    factory_remaining 4   (fraction of stack remaining)
    collected         8   (2×4, normalised factory value)
    penalty_pile     16   (2×4×2, public composition)
    draw_pile_size    1

Architecture:
  flat_enc  : Linear(85, 128) → LayerNorm → ReLU
  fusion    : Linear(128+32+32, 256) → LayerNorm → ReLU
              → Linear(256, 128) → LayerNorm → ReLU
  actor     : Linear(128, 108)   [masked softmax]
  critic    : Linear(128, 1)
```

---

## Async Batch Stepping

The vectorised environment uses Python generators to decouple the game loop from
neural-network evaluation:

```
Old (sequential):
  for game in N_games:
      while opponent_turn:
          action = NN(obs)          ← N × O separate NN calls (batch=1 each)
          game.step(action)

New (generator / batched):
  gens = [game.step_gen(agent_action) for game in N_games]
  while any active:
      pending_obs = [next(gen) for gen in active_gens]   ← collect from all games
      actions = NN(stack(pending_obs))                   ← ONE batched NN call
      [gen.send(action) for gen, action in zip(...)]     ← send back
```

`step_gen()` is a generator on `SinglePlayerEcoEnv`. The vec env drives all generators
in lockstep so each "round" of opponent play costs one forward pass regardless of N.

---

## Running the Code

### Requirements

```bash
pip install numpy torch flask tyro
# TensorBoard (optional but recommended):
pip install tensorboard
# Weights & Biases (optional):
pip install wandb
```

### 1 — Run tests

```bash
python -m pytest eco_tests.py -v
```

### 2 — Train (self-play, default)

```bash
python eco_ppo.py
```

Checkpoints are saved to `model/eco_<step>.pkt` and `model/eco_latest.pkt` every
500 000 steps. TensorBoard logs go to `runs/`.

```bash
tensorboard --logdir runs/
```

### 3 — Training flags (all inherited from `Args` dataclass)

#### Core

| Flag | Default | Description |
|------|---------|-------------|
| `--num_players` | `2` | Number of players (2–5) |
| `--opponent_mode` | `self_play` | `self_play` / `random` / `mixed` |
| `--total_timesteps` | `50_000_000` | Total environment steps |
| `--num_envs` | `128` | Parallel environments |
| `--num_steps` | `32` | Rollout steps per env per update |
| `--seed` | `1` | Global random seed |

#### Optimisation

| Flag | Default | Description |
|------|---------|-------------|
| `--learning_rate` | `2.5e-4` | Adam learning rate |
| `--anneal_lr` | `False` | Linearly decay LR over training |
| `--gamma` | `1.0` | Discount factor (1.0 = undiscounted) |
| `--gae_lambda` | `1.0` | GAE λ (keep 1.0 when gamma=1.0) |
| `--num_minibatches` | `4` | Mini-batches per PPO update |
| `--update_epochs` | `4` | PPO epochs per rollout |
| `--clip_coef` | `0.2` | PPO clip ε |
| `--ent_coef` | `0.01` | Entropy bonus coefficient |
| `--vf_coef` | `0.5` | Value loss coefficient |
| `--max_grad_norm` | `0.5` | Gradient clipping |
| `--target_kl` | `0.01` | Early-stop KL threshold |

#### Reward shaping

| Flag | Default | Description |
|------|---------|-------------|
| `--reward_shaping_scale` | `1.0` | Scale for immediate factory-card rewards |

Set to `0.0` to train on terminal reward only. Values 0.5–2.0 provide useful
intermediate signal during exploration.

#### Logging / saving

| Flag | Default | Description |
|------|---------|-------------|
| `--log_interval` | `10_000` | Print training stats every N steps |
| `--save_interval` | `500_000` | Save checkpoint every N steps |
| `--track` | `False` | Enable W&B logging |
| `--wandb_project_name` | `ppo-eco` | W&B project |
| `--wandb_entity` | `None` | W&B entity (team) |
| `--exp_name` | `eco_ppo` | Run name prefix |

#### Hardware

| Flag | Default | Description |
|------|---------|-------------|
| `--cuda` | `True` | Use GPU if available |
| `--torch_deterministic` | `True` | cuDNN determinism |

### 4 — Common training recipes

```bash
# Quick sanity check (random opponents, 1M steps, no GPU)
python eco_ppo.py --opponent_mode random --total_timesteps 1_000_000 --cuda

# Self-play, 3 players, W&B logging
python eco_ppo.py --num_players 3 --track --wandb_project_name r-eco --cuda

# Mixed opponents, more envs, shaping disabled
python eco_ppo.py --opponent_mode mixed --num_envs 256 --reward_shaping_scale 0.0

# Reproducible run
python eco_ppo.py --seed 42 --torch_deterministic True

# Resume / continue training (just start again — checkpoints accumulate)
python eco_ppo.py --seed 99
```

### 5 — Play in the browser

```bash
# Serve both Hearts and R-öko
python server.py

# With a trained R-öko model
python server.py --model-dir model --port 5000
```

Open:
- `http://localhost:5000/`    — Hearts
- `http://localhost:5000/eco` — R-öko

Select opponent type **"Trained AI"** in the browser once `model/eco_latest.pkt` exists.

### 6 — TensorBoard metrics

| Tag | Description |
|-----|-------------|
| `charts/SPS` | Environment steps per second |
| `losses/pg_loss` | PPO policy gradient loss |
| `losses/v_loss` | Value function loss |
| `losses/entropy` | Policy entropy |
| `losses/approx_kl` | Approximate KL divergence |
| `losses/clipfrac` | Fraction of clipped PPO updates |
| `benchmark/vs_random/mean_score` | Agent score vs random opponent |
| `benchmark/vs_random/mean_reward` | Normalised reward vs random opponent |

---

## Architecture Diagram

```
Training loop (eco_ppo.py)
│
├── VecSinglePlayerEcoEnv  (eco_vec_env.py)
│   │  N parallel SinglePlayerEcoEnv instances
│   │
│   ├── step(agent_actions, batch_opponent_fn)
│   │     │
│   │     ├── start N step_gen() generators
│   │     ├── prime all → collect pending opponent obs
│   │     ├── batch_opponent_fn(stacked_obs) → N actions  ← single NN forward pass
│   │     └── send actions back via gen.send()
│   │
│   └── SinglePlayerEcoEnv  (eco_obs_encoder.py)
│         │
│         ├── step_gen(action)   ← generator, yields (obs, mask) per opponent step
│         ├── _encode_for()      ← EcoPyTreeObs from EcoState
│         └── EcoEnv             ← pure game logic (eco_env.py)
│
└── EcoAgent  (eco_ppo.py)
      ├── player_emb + phase_emb  (embedding lookup)
      ├── flat_enc                (85 float features → 128)
      ├── fusion MLP              (128+64 → 256 → 128)
      ├── actor_head              (128 → 108 actions)
      └── critic_head             (128 → 1 value)

Server (server.py)
  /          → Hearts  (static/index.html)
  /eco       → R-öko   (static/eco_index.html)
  /api/eco/new_game   POST → EcoGameSession
  /api/eco/play       POST → EcoGameSession.human_action()
```
