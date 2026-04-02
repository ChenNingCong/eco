#!/bin/bash
# Baseline: no opponent penalty (original reward)
python eco_ppo.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0 \
    --reward-shaping-scale 1.0 \
    --vf-coef 1.0 \
    --total-timesteps 2000000 \
    --model-dir model/baseline \
    --exp-name baseline \
    --track
