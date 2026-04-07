#!/bin/bash
# Lost Cities: decomposed actions (106) + max lanes 5 + [0,1] reward + gamma=0.999.
# Testing if removing negative reward + discounting fixes the ddraw explosion.
# Uses 2048 envs (config that previously exploded with +1/-1 gamma=1.0).
# Entropy 0.1→0.01 exp over 1M.
python -u -m game.lc.train \
    --opponent-mode self_play \
    --vf-coef 1.0 \
    --ent-coef 0.1 \
    --ent-coef-end 0.01 \
    --ent-anneal-steps 1000000 \
    --ent-anneal-mode exponential \
    --gamma 0.999 \
    --gae-lambda 0.85 \
    --lstm-hidden 128 \
    --num-envs 2048 \
    --num-steps 64 \
    --total-timesteps 50000000 \
    --new-color-penalty 20 \
    --max-lanes 5 \
    --decompose-actions \
    --zero-one-reward \
    --model-dir model/lc_decompose_lanes5_01reward \
    --exp-name lc_decompose_lanes5_01reward \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
