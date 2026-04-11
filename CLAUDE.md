# Eco — Multi-game RL Framework

## Project Overview
RL training framework for board games (Eco, Ticket to Ride, Lost Cities). Current focus: **Lost Cities (LC)** NFSP training and DQN reward ablations.

## Quick Reference

### Key Paths
- **LC engine:** `game/lc/engine.py` (env, reward modes, action masking)
- **LC PPO agent:** `game/lc/agent.py` (LSTM/FF, 600 or 106 action space)
- **LC DQN agent:** `game/lc/dqn_agent.py` (Q-network, 256 hidden default)
- **LC training:** `game/lc/train.py` (PPO), `train_dqn.py` (DQN), `train_nfsp.py` (NFSP)
- **LC server:** `game/lc/server.py` (Flask UI for human vs AI)
- **Benchmarking:** `scripts/bench_agents.py`
- **Equilibrium notes:** `equilibrum.md` (penalty annealing + NFSP algorithm)

### Trained Models (as of 2026-04-07)
| Directory | Type | Steps | Notes |
|-----------|------|-------|-------|
| `model/lc_50m_h512` | PPO LSTM h512 | 50M | Baseline, score ~40 vs random |
| `model/lc_50m_ff` | PPO FF h512 | 50M | FF baseline |
| `model/lc_dqn_dense` | DQN dense reward | 50M | Dense reward, 256 hidden |
| `model/lc_dqn_dense_dd5` | DQN dense+dd5 | 50M | Dense reward, max_discard_draws=5 |
| `model/lc_dqn_dense_smallbuf` | DQN dense small buf | 50M | Dense reward, smaller buffer |
| `model/lc_ppo_from_dqn_h512_noent` | PPO from DQN init | ~24M | h512, ent_coef=0.0 |
| `model/lc_ppo_from_dqn_h512_ent01` | PPO from DQN init | ~24M | h512, ent_coef=0.01 |
| `model/lc_nfsp_v1` | NFSP | in progress | eta=0.1, dense reward, dd=50 |

## Current Work

### 1. DQN Reward Ablations (first priority)
Goal: Systematic ablation of reward functions for DQN training. Get solid results here before moving to NFSP.

Reward modes (set in engine, only one active at a time):
- `--dense-reward`: per-step (current_score - prev_score) / 30
- `--score-diff-reward`: terminal (own - opponent) / 30
- `--zero-one-reward`: terminal 1 if win else 0
- `--raw-score-reward`: terminal own_score / 30
- Default (no flag): +1 win / -1 loss / 0 draw

Planned combinations to test: dense, dense+score_diff, score_diff alone, default win/loss.

### 2. NFSP Training (after reward ablations are solid)
Goal: Train NFSP agent to approximate Nash equilibrium, solving strategy cycling in self-play.

Launch command:
```bash
python -u -m game.lc.train_nfsp \
  --exp-name lc_nfsp_v1 --dense-reward --max-discard-draws 50 \
  --eta 0.1 --rl-buffer-size 200000 --sl-buffer-size 200000 \
  --total-timesteps 20000000 --num-envs 128 \
  --model-dir model/lc_nfsp_v1 --save-interval 819200 \
  --track --wandb-project-name ppo-eco --seed 1
```


## Critical Rules
- **Always set `--max-discard-draws 50`** for DQN/NFSP self-play — without it, agents deadlock in discard-draw cycles.
- **PPO uses `--max-discard-draws 5`** (different from DQN).
- **Kill ALL old processes before relaunching** — OOM kills happen silently on this 16GB machine.
- **Ablation configs must exactly match the reference run**, only varying the ablation variable.
- **wandb project:** `ppo-eco` for all LC experiments.
- **Action space:** 600 flat actions (50 cards x 2 types x 6 draw sources). Decomposed (106) exists but flat is more robust.
- **Observation dim:** 303 floats. Q-network/encoder: 2-layer MLP with LayerNorm+ReLU.

## LC Server
```bash
python -m game.lc.server \
  --dqn-dir model/lc_dqn_dense_dd5 model/lc_dqn_dense model/lc_dqn_dense_smallbuf \
  --model-dir model/lc_ppo_from_dqn_h512_noent model/lc_ppo_from_dqn_h512_ent01 model/lc_50m_h512 \
  --ppo-hidden-dim 512 --port 5002
```
