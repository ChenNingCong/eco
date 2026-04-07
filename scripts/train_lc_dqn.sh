#!/bin/bash
# Lost Cities DQN baseline: flat 600-action, vs random opponent, 50M steps.
# Comparable to lc_baseline_50m_ff (same game config, same opponent, FF architecture).
#
# With 128 envs, each step = 128 transitions. Settings:
#   batch_size=2048: ~0.2% of 1M buffer, good gradient signal
#   train_frequency=1: train every env step (128 transitions per step)
#   grad_steps_per_train=1: 1 gradient update per 128 transitions
python -u -m game.lc.train_dqn \
    --opponent-mode self_play \
    --learning-rate 1e-4 \
    --num-envs 128 \
    --buffer-size 1000000 \
    --gamma 1.0 \
    --batch-size 2048 \
    --start-e 1.0 \
    --end-e 0.05 \
    --exploration-fraction 0.10 \
    --learning-starts 10000 \
    --train-frequency 1 \
    --target-network-frequency 1000 \
    --total-timesteps 50000000 \
    --new-color-penalty 20 \
    --hidden-dim 256 \
    --max-discard-draws 50 \
    --model-dir model/lc_dqn \
    --exp-name lc_dqn \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
