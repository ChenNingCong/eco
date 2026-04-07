#!/bin/bash
# Lost Cities DQN ablation: dense reward (per-step score delta for current player).
# Based on loweps (start_e=0.1) + dense_reward.
python -u -m game.lc.train_dqn \
    --opponent-mode self_play \
    --learning-rate 1e-4 \
    --num-envs 128 \
    --buffer-size 1000000 \
    --gamma 1.0 \
    --batch-size 2048 \
    --start-e 0.1 \
    --end-e 0.05 \
    --exploration-fraction 0.10 \
    --learning-starts 10000 \
    --train-frequency 1 \
    --target-network-frequency 1000 \
    --total-timesteps 50000000 \
    --new-color-penalty 20 \
    --hidden-dim 256 \
    --max-discard-draws 50 \
    --dense-reward \
    --model-dir model/lc_dqn_dense \
    --exp-name lc_dqn_dense \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
