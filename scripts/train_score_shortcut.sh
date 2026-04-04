#!/bin/bash
# Score shortcut: shallow path from scores to critic, no reward shaping
python eco_ppo.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0 \
    --reward-shaping-scale 0 \
    --vf-coef 1.0 \
    --relative-seat \
    --score-shortcut \
    --total-timesteps 2000000 \
    --model-dir model/score_shortcut \
    --exp-name score_shortcut \
    --track
