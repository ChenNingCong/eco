#!/bin/bash
# Ablation 4: LSTM h=128 vs random opponent — is co-adaptation the problem?
python -u eco_ppo_lstm.py \
    --num-players 3 \
    --opponent-mode random \
    --opponent-penalty 0 \
    --reward-shaping-scale 0 \
    --vf-coef 1.0 \
    --ent-coef 0.1 \
    --ent-coef-end 0.01 \
    --ent-anneal-steps 40000 \
    --gae-lambda 0.85 \
    --relative-seat \
    --lstm-hidden 128 \
    --total-timesteps 10000000 \
    --model-dir model/ablation_vs_random \
    --exp-name ablation_vs_random \
    --seed 1 \
    --track
