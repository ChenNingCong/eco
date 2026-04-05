#!/bin/bash
# Ablation: old FF arch + sequential minibatch (isolate network vs minibatch effect)
# Same as ablation_ff but with --sequential-mb to match LSTM minibatch strategy
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
    --sequential-mb \
    --total-timesteps 10000000 \
    --model-dir model/ablation_ff_seqmb \
    --exp-name ablation_ff_seqmb \
    --seed 1 \
    --track
