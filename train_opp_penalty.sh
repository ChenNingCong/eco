#!/bin/bash
# With opponent penalty: r = my_r - 0.5 * max(opp_r)
python eco_ppo.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0.5 \
    --reward-shaping-scale 1.0 \
    --vf-coef 1.0 \
    --total-timesteps 2000000 \
    --model-dir model/opp_penalty \
    --exp-name opp_penalty \
    --track
