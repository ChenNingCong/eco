#!/bin/bash
# Ablation: scale_route_score=0.1 + scale_dest_penalty=0.1 (combined).
# Routes worth 10x less AND destination failure costs 10x less for reward.
# Entropy 0.1→0.01 exp over 1M, 10M total.
python -u -m game.ttr.train \
    --num-players 2 \
    --opponent-mode self_play \
    --vf-coef 1.0 \
    --ent-coef 0.1 \
    --ent-coef-end 0.01 \
    --ent-anneal-steps 1000000 \
    --ent-anneal-mode exponential \
    --gae-lambda 0.85 \
    --lstm-hidden 256 \
    --num-envs 128 \
    --num-steps 64 \
    --total-timesteps 10000000 \
    --scale-route-score 0.1 \
    --scale-dest-penalty 0.1 \
    --model-dir model/ttr_shape \
    --exp-name shape_both01 \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
