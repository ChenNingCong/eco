#!/bin/bash
# Terminal-only reward + GAE lambda=0.95 (bootstrapped, lower variance)
python eco_ppo.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0 \
    --reward-shaping-scale 0 \
    --vf-coef 1.0 \
    --gae-lambda 0.95 \
    --relative-seat \
    --total-timesteps 2000000 \
    --model-dir model/gae95 \
    --exp-name gae95 \
    --track
