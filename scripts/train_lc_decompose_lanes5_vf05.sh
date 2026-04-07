#!/bin/bash
# Lost Cities: decomposed actions (106) + max lanes 5 + gae 0.85 + vf-coef 0.5, 50M steps.
# Entropy 0.1→0.01 exp over 1M.
python -u -m game.lc.train \
    --opponent-mode self_play \
    --vf-coef 0.5 \
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
    --model-dir model/lc_decompose_lanes5_vf05 \
    --exp-name lc_decompose_lanes5_vf05 \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
