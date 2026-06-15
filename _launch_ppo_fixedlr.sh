#!/bin/bash
# SIMPLE LC PPO self-play: feedforward (NO LSTM), trained FROM SCRATCH (no DQN bootstrap).
# Reward: DENSE zero-sum (own-opp) score-diff = (Δown-Δopp)/30 per step.
# NO discard limit (max_dd default 0 = unlimited; the cap was a DQN-only fix for
# self-play loops). Adaptive KL OFF; FIXED base LR linearly annealed to 0.
# Usage:  GPU=N bash _launch_ppo_fixedlr.sh <lr> <tag>
set -u
LR="$1"; TAG="$2"
source ~/programs/anaconda3/etc/profile.d/conda.sh && conda activate minecraft
cd /home/zzhang18/nchen3/eco
export CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "[$TAG] lr=$LR gpu=$CUDA_VISIBLE_DEVICES start $(date '+%F %T')"
python -u -m game.lc.train \
  --no-lstm \
  --dense-reward --score-diff-reward --opponent-mode self_play \
  --hidden-dim 512 --ent-coef 0.0 --critic-warmup-steps 100000 \
  --learning-rate "$LR" --anneal-lr --target-kl None \
  --gae-lambda 0.95 --gamma 1 --opponent-sync-interval 50 \
  --num-envs 128 --num-steps 32 \
  --num-minibatches 4 --update-epochs 4 --clip-coef 0.2 \
  --total-timesteps 50000000 --wandb-project-name ppo-eco --seed 1 --track \
  --model-dir "model/lc_ppo_ffdiff_${TAG}" \
  --exp-name "lc_ppo_ffdiff_${TAG}"
echo "[$TAG] done $(date '+%F %T') exit=$?"
