#!/bin/bash
# Train Ticket to Ride with PPO+LSTM.
# TTR has larger obs/action space than R-Öko, so we use wider nets
# and more environments. Hyperparams adapted from best R-Öko run.
python -u -m game.ttr.train \
    --num-players 2 \
    --opponent-mode self_play \
    --vf-coef 1.0 \
    --ent-coef 0.1 \
    --ent-coef-end 0.01 \
    --ent-anneal-steps 100000 \
    --gae-lambda 0.85 \
    --lstm-hidden 256 \
    --num-envs 128 \
    --num-steps 64 \
    --total-timesteps 10000000 \
    --model-dir model/ttr \
    --exp-name ttr_ablation \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
