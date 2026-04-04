#!/bin/bash
# With opponent penalty + lower value loss coefficient
python eco_ppo.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0.5 \
    --reward-shaping-scale 1.0 \
    --vf-coef 0.1 \
    --total-timesteps 2000000 \
    --model-dir model/opp_penalty_lowvf \
    --exp-name opp_penalty_lowvf \
    --track
