---
title: R-oko
emoji: ♻
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# R-oko: PPO Self-Play for a Multiplayer Card Game

Train and play against an RL agent for the R-oko (R-öko) card game using PPO with self-play.

**Live demo:** [huggingface.co/spaces/NingcongChen/r-oko](https://huggingface.co/spaces/NingcongChen/r-oko)
**Training report:** [wandb report](https://wandb.ai/ningcong-chen/ppo-eco/reports/R-oko-PPO-Training:-Ablation-Study--VmlldzoxNjQwNTU3MQ==)

## Setup

```bash
# Clone
git clone https://github.com/ChenNingCong/eco.git
cd eco

# Install dependencies (Python 3.10+)
pip install torch numpy tyro flask wandb
```

## Train

```bash
# Train with best config (entropy annealing + GAE=0.85, 50M samples, ~6 hours on 1 GPU)
bash train_ent_gae_50m.sh

# Or a quick 10M run (~1 hour)
bash train_ent_gae_10m.sh
```

Training logs to [Weights & Biases](https://wandb.ai). Remove `--track` from the script to disable.

Checkpoints are saved to `model/<exp_name>/` every ~80k samples.

## Play Against the Agent

```bash
# Start the web server (loads checkpoint from model dir)
python server.py --model-dir model/ent_gae_50m --port 5000
```

Open `http://localhost:5000` in your browser. You can choose your seat and number of players.

## Run Tests

```bash
python eco_tests.py
```

## Project Structure

```
eco_env.py          # Core game logic (rules, state, actions)
eco_obs_encoder.py  # Observation encoding + single-player env wrapper
eco_vec_env.py      # Vectorized env with generator-based batched opponent inference
eco_ppo.py          # PPO training loop + agent network
server.py           # Flask web server for human vs AI play
static/             # Web UI
train_*.sh          # Training scripts for different configs
```
