#!/bin/bash
# Lost Cities: decomposed actions (106) + max lanes 5 + pure feedforward (no LSTM), 50M steps.
# Architecture: 2-layer encoder + 2-layer middle MLP + 2-layer trunks.
# Entropy 0.1→0.01 exp over 1M.
python -u -m game.lc.train \
    --opponent-mode self_play \
    --vf-coef 1.0 \
    --ent-coef 0.1 \
    --ent-coef-end 0.01 \
    --ent-anneal-steps 1000000 \
    --ent-anneal-mode exponential \
    --gae-lambda 0.85 \
    --lstm-hidden 128 \
    --num-envs 128 \
    --num-steps 64 \
    --total-timesteps 50000000 \
    --new-color-penalty 20 \
    --max-lanes 5 \
    --decompose-actions \
    --no-lstm \
    --model-dir model/lc_decompose_lanes5_ff \
    --exp-name lc_decompose_lanes5_ff \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
