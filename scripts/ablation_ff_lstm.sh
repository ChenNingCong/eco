#!/bin/bash
# Ablation: old FF arch + LSTM concat (recurrence without capacity loss)
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
    --network-arch ff_lstm \
    --lstm-hidden 128 \
    --network-cell-clip 5.0 \
    --total-timesteps 10000000 \
    --model-dir model/ablation_ff_lstm \
    --exp-name ablation_ff_lstm \
    --seed 1 \
    --track
