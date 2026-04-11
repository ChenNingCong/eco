#!/bin/bash
# Lost Cities DQN vs random: dense + score-diff reward.
# Per-step delta(own-opp)/30 at terminal; delta(own)/30 mid-game.
# Episode total = (own_score - opp_score) / 30.
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
    --dense-reward \
    --score-diff-reward \
    --model-dir model/lc_dqn_vs_random_dense_scorediff \
    --exp-name lc_dqn_vs_random_dense_scorediff \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
