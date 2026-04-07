#!/bin/bash
# Lost Cities baseline: penalty=-20 (default).
# Entropy 0.1→0.01 exp over 1M, 10M total.
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
    --total-timesteps 10000000 \
    --new-color-penalty 20 \
    --model-dir model/lc \
    --exp-name lc_baseline \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
