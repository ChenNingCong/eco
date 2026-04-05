#!/bin/bash
# Ablation 3: Wider LSTM h=256 (no bottleneck) — is 128 too narrow?
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
    --lstm-hidden 256 \
    --total-timesteps 10000000 \
    --model-dir model/ablation_h256 \
    --exp-name ablation_h256 \
    --seed 1 \
    --track
