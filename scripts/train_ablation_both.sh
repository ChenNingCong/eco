#!/bin/bash
# Ablation: entropy annealing 0.1 → 0.01 + GAE=0.85, 10M samples
python -u eco_ppo.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0 \
    --reward-shaping-scale 0 \
    --vf-coef 1.0 \
    --ent-coef 0.1 \
    --ent-coef-end 0.01 \
    --ent-anneal-steps 40000 \
    --gae-lambda 0.85 \
    --relative-seat \
    --total-timesteps 10000000 \
    --model-dir model/abl_both \
    --exp-name abl_both \
    --track
