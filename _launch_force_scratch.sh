#!/bin/bash
# From-scratch PPO self-play + CURRICULUM EXPLORATION (forced investment).
# Tests the hypothesis: a from-scratch agent never explores investment (P(invest)->0,
# absorbing). Forcing early-game investment plays (preferring colors already invested in,
# to build 2-3 stacks) + truncated-IS off-policy handling should teach it the value of
# stacks and let it reach high scores WITHOUT any DQN/BC init.
#
# Forcing is training-only (eval unaffected) and ANNEALS to 0 over the first half of
# training so it does not pollute the final equilibrium.
#
# Usage:  GPU=N bash _launch_force_scratch.sh [force_prob] [tag]   (defaults 0.5, ent01)
set -u
FP="${1:-0.5}"; TAG="${2:-ent01}"
source ~/programs/anaconda3/etc/profile.d/conda.sh && conda activate minecraft
cd /home/zzhang18/nchen3/eco
export CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "[force_scratch:$TAG] force_prob=$FP gpu=$CUDA_VISIBLE_DEVICES start $(date '+%F %T')"
python -u -m game.lc.train \
  --no-lstm --hidden-dim 512 --ent-coef 0.01 \
  --score-diff-reward --opponent-mode self_play \
  --max-discard-draws 5 --critic-warmup-steps 0 \
  --force-explore-prob "$FP" --force-explore-until 8 \
  --force-explore-anneal-frac 0.5 --force-explore-clip 5 \
  --learning-rate 2.5e-4 --anneal-lr --target-kl None \
  --gae-lambda 0.95 --gamma 1 --opponent-sync-interval 50 \
  --num-envs 128 --num-steps 32 \
  --num-minibatches 4 --update-epochs 4 --clip-coef 0.2 \
  --total-timesteps 50000000 --wandb-project-name ppo-eco --seed 1 --track \
  --model-dir "model/lc_ppo_force_scratch_${TAG}" \
  --exp-name "lc_ppo_force_scratch_${TAG}"
echo "[force_scratch:$TAG] done $(date '+%F %T') exit=$?"
