#!/bin/bash
# Train R-Öko with best-performing settings (ablation_ff_lstm config).
# 3 players, self-play, no reward shaping, entropy 0.1→0.01, GAE λ=0.85
python -u -m game.r_eco.train \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0 \
    --reward-shaping-scale 0 \
    --vf-coef 1.0 \
    --ent-coef 0.1 \
    --ent-coef-end 0.01 \
    --ent-anneal-steps 40000 \
    --gae-lambda 0.85 \
    --lstm-hidden 128 \
    --total-timesteps 10000000 \
    --model-dir model/r_eco \
    --exp-name r_eco_lstm \
    --seed 1 \
    --track
