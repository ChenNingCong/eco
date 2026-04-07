#!/bin/bash
# Lost Cities DQN ablation: dense reward + small replay buffer (100K instead of 1M).
# Same as train_lc_dqn_dense.sh except --buffer-size 100000.
python -u -m game.lc.train_dqn \
    --opponent-mode self_play \
    --learning-rate 1e-4 \
    --num-envs 128 \
    --buffer-size 100000 \
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
    --hidden-dim 256 \
    --max-discard-draws 50 \
    --dense-reward \
    --model-dir model/lc_dqn_dense_smallbuf \
    --exp-name lc_dqn_dense_smallbuf \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
