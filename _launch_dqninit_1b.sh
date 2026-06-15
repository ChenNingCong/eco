#!/bin/bash
# Continue the BC-init -> PPO self-play recipe for 1B STEPS (the 50M run was good but short).
# Warm-starts from the 50M-FINAL checkpoint (the run's best: 73% vs heuristic, score 35.5,
# 24.9 self-play) with LR at 1/10 (2.5e-5) so it PRESERVES that settled/sharp policy and
# refines gently, instead of the 2.5e-4 jump that un-sharpened it (the LR-decay artifact).
# save-interval 10M -> ~100 checkpoints (not 12k) to keep disk sane.
#
# Usage:  GPU=N bash _launch_dqninit_1b.sh
set -u
source ~/programs/anaconda3/etc/profile.d/conda.sh && conda activate minecraft
cd /home/zzhang18/nchen3/eco
export CUDA_VISIBLE_DEVICES=${GPU:-0}
INIT=model/lc_ppo_dqninit_final50m/latest.pkt
echo "[dqninit_1b] warm-start from $INIT (lr 2.5e-5)  gpu=$CUDA_VISIBLE_DEVICES  start $(date '+%F %T')"
python -u -m game.lc.train \
  --pretrained "$INIT" \
  --no-lstm --hidden-dim 512 --ent-coef 0.01 \
  --score-diff-reward --opponent-mode self_play \
  --max-discard-draws 5 --critic-warmup-steps 0 \
  --learning-rate 2.5e-5 --anneal-lr --target-kl None \
  --gae-lambda 0.95 --gamma 1 --opponent-sync-interval 50 \
  --num-envs 128 --num-steps 32 --num-minibatches 4 --update-epochs 4 --clip-coef 0.2 \
  --save-interval 10000000 \
  --total-timesteps 1000000000 --wandb-project-name ppo-eco --seed 1 --track \
  --model-dir model/lc_ppo_dqninit_1b --exp-name lc_ppo_dqninit_1b
echo "[dqninit_1b] done $(date '+%F %T') exit=$?"
