#!/bin/bash
# Absolute seat encoding, no reward shaping
python eco_ppo.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0 \
    --reward-shaping-scale 0 \
    --vf-coef 1.0 \
    --no-relative-seat \
    --total-timesteps 2000000 \
    --model-dir model/absolute_seat \
    --exp-name absolute_seat \
    --track
