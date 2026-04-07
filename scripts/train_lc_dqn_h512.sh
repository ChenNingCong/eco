#!/bin/bash
# Lost Cities DQN ablation: hidden_dim=512 (larger FF model).
# Same as train_lc_dqn.sh except --hidden-dim 512.
python -u -m game.lc.train_dqn \
    --opponent-mode self_play \
    --learning-rate 1e-4 \
    --num-envs 128 \
    --buffer-size 1000000 \
    --gamma 1.0 \
    --batch-size 2048 \
    --start-e 1.0 \
    --end-e 0.05 \
    --exploration-fraction 0.10 \
    --learning-starts 10000 \
    --train-frequency 1 \
    --target-network-frequency 1000 \
    --total-timesteps 50000000 \
    --new-color-penalty 20 \
    --hidden-dim 512 \
    --max-discard-draws 50 \
    --model-dir model/lc_dqn_h512 \
    --exp-name lc_dqn_h512 \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
