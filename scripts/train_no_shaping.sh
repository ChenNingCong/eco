#!/bin/bash
# No reward shaping — only terminal +1/-1
python eco_ppo.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0 \
    --reward-shaping-scale 0 \
    --vf-coef 1.0 \
    --relative-seat \
    --total-timesteps 2000000 \
    --model-dir model/no_shaping \
    --exp-name no_shaping \
    --track
