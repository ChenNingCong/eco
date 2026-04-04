#!/bin/bash
# Relative seat encoding (current_player always 0)
python eco_ppo.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0.5 \
    --reward-shaping-scale 1.0 \
    --vf-coef 1.0 \
    --relative-seat \
    --total-timesteps 2000000 \
    --model-dir model/relative_seat \
    --exp-name relative_seat \
    --track
