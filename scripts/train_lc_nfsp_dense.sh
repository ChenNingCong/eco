#!/bin/bash
# Lost Cities NFSP: dense reward, matching the DQN vs-random reward design.
# Self-play (RL head vs average policy), so discard limit 50 still needed to prevent cycling.
# RL/SL buffers at 1M/2M (NFSP defaults) for stable convergence.
# Target: >40 mean score vs random on both RL and SL heads.
python -u -m game.lc.train_nfsp \
    --exp-name lc_nfsp_dense \
    --dense-reward \
    --max-discard-draws 50 \
    --eta 0.1 \
    --rl-buffer-size 1000000 \
    --sl-buffer-size 2000000 \
    --rl-batch-size 256 \
    --sl-batch-size 256 \
    --rl-lr 1e-4 \
    --sl-lr 1e-4 \
    --rl-train-frequency 4 \
    --sl-train-frequency 4 \
    --rl-target-network-frequency 1000 \
    --rl-learning-starts 10000 \
    --rl-start-e 1.0 \
    --rl-end-e 0.05 \
    --rl-exploration-fraction 0.10 \
    --total-timesteps 50000000 \
    --num-envs 128 \
    --hidden-dim 256 \
    --new-color-penalty 20 \
    --model-dir model/lc_nfsp_dense \
    --save-interval 819200 \
    --track \
    --wandb-project-name ppo-eco \
    --seed 1
