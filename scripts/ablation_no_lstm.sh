#!/bin/bash
# Ablation 2: Disable LSTM (zero hidden every step) — is LSTM the problem?
python -u eco_ppo_lstm.py \
    --num-players 3 \
    --opponent-mode self_play \
    --opponent-penalty 0 \
    --reward-shaping-scale 0 \
    --vf-coef 1.0 \
    --ent-coef 0.1 \
    --ent-coef-end 0.01 \
    --ent-anneal-steps 40000 \
    --gae-lambda 0.85 \
    --relative-seat \
    --lstm-hidden 128 \
    --disable-lstm \
    --total-timesteps 10000000 \
    --model-dir model/ablation_no_lstm \
    --exp-name ablation_no_lstm \
    --seed 1 \
    --track
