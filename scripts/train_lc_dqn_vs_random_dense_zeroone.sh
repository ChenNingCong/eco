#!/bin/bash
# Lost Cities DQN vs random: dense + zero-one reward.
# Per-step delta(own)/30 as shaping + terminal (1 if win else 0).
# Episode total = own_score/30 + zero_one_bonus.
python -u -m game.lc.train_dqn \
    --opponent-mode random \
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
    --hidden-dim 512 \
    --max-discard-draws 50 \
    --dense-reward \
    --zero-one-reward \
    --model-dir model/lc_dqn_vs_random_dense_zeroone \
    --exp-name lc_dqn_vs_random_dense_zeroone \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
