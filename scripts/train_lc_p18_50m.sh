#!/bin/bash
# Lost Cities ablation 50M: penalty=-18.
# Entropy 0.1→0.01 exp over 1M, 50M total.
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
    --new-color-penalty 18 \
    --model-dir model/lc_p18_50m \
    --exp-name lc_p18_50m \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
