#!/bin/bash
# Entropy annealing: 0.05 → 0.01 over first 40k steps, then hold at 0.01
python -u eco_ppo.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0 \
    --reward-shaping-scale 0 \
    --vf-coef 1.0 \
    --ent-coef 0.1 \
    --ent-coef-end 0.01 \
    --ent-anneal-steps 40000 \
    --relative-seat \
    --total-timesteps 2000000 \
    --model-dir model/ent_anneal \
    --exp-name ent_anneal \
    --track
