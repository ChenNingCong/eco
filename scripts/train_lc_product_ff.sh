#!/bin/bash
# Lost Cities product action heads (card×type×draw factored logits), FF, 50M.
# Matches lc_baseline_50m_ff config but with --product-actions.
python -u -m game.lc.train \
    --opponent-mode self_play \
    --vf-coef 1.0 \
    --ent-coef 0.1 \
    --ent-coef-end 0.01 \
    --ent-anneal-steps 1000000 \
    --ent-anneal-mode exponential \
    --gae-lambda 0.85 \
    --lstm-hidden 128 \
    --num-envs 128 \
    --num-steps 64 \
    --total-timesteps 50000000 \
    --new-color-penalty 20 \
    --no-lstm \
    --product-actions \
    --model-dir model/lc_product_ff \
    --exp-name lc_product_ff \
    --wandb-project-name ppo-eco \
    --seed 1 \
    --track
