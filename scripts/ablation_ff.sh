#!/bin/bash
# Ablation: old feedforward architecture (no LSTM, 2-layer trunks, random minibatch)
# Should reproduce ent_gae_10m results (~100% vs random)
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
    --network-arch ff \
    --total-timesteps 10000000 \
    --model-dir model/ablation_ff \
    --exp-name ablation_ff \
    --seed 1 \
    --track
