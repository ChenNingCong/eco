#!/bin/bash
# Ablation: clip LSTM cell state to [-5, 5] (OpenAI Five style)
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
    --network-cell-clip 5.0 \
    --total-timesteps 10000000 \
    --model-dir model/ablation_cell_clip \
    --exp-name ablation_cell_clip \
    --seed 1 \
    --track
