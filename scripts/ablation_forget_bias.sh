#!/bin/bash
# Ablation: forget gate bias = 1.0 (Jozefowicz 2015)
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
    --network-forget-gate-bias 1.0 \
    --total-timesteps 10000000 \
    --model-dir model/ablation_forget_bias \
    --exp-name ablation_forget_bias \
    --seed 1 \
    --track
