#!/bin/bash
# Lost Cities DQN vs random ablation: default reward (+1 win / -1 loss / 0 draw).
python -u -m game.lc.train_dqn \
    --opponent-mode random \
    --learning-rate 1e-4 \
    --num-envs 128 \
    --buffer-size 1000000 \
    --gamma 1.0 \
    --batch-size 2048 \
    --start-e 0.1 \
    --end-e 0.05 \
    --exploration-fraction 0.10 \
    --learning-starts 10000 \
    --train-frequency 1 \
    --target-network-frequency 1000 \
    --total-timesteps 50000000 \
    --new-color-penalty 20 \
    --hidden-dim 512 \
    --max-discard-draws 50 \
    --model-dir model/lc_dqn_vs_random_default \
    --exp-name lc_dqn_vs_random_default \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
