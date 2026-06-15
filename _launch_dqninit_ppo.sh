#!/bin/bash
# Replicate lc_ppo_dqninit_ent01: initialize PPO from a DQN via behavioral cloning,
# then PPO self-play finetune.
#
#   Stage 1 (BC): imitate_dqn.py clones the DQN greedy policy into a no-LSTM PPO agent.
#   Stage 2 (PPO): train.py --pretrained <bc.pkt> continues with self-play, ent-coef 0.01.
#
# Source DQN : lc_dqn_vs_random_dense_dd5/final.pt  (hidden=512, dense OWN-score, dd5)
#              = the STRONGEST validated teacher: beats heuristic 65%, inv 3.2, lanes 2.6,
#              i.e. already PAST the investment barrier that from-scratch PPO can't cross.
# Arch       : feedforward (--no-lstm), hidden 512
# Finetune   : TERMINAL score-diff (zero-sum competitive target) + dd5 cap. Terminal (not
#              dense) so the per-step investment barrier doesn't erode the BC-seeded investing.
#
# Usage:  GPU=N bash _launch_dqninit_ppo.sh [lr] [tag]   (defaults: lr=2.5e-4 tag=ent01)
set -u
LR="${1:-2.5e-4}"; TAG="${2:-ent01}"
source ~/programs/anaconda3/etc/profile.d/conda.sh && conda activate minecraft
cd /home/zzhang18/nchen3/eco
export CUDA_VISIBLE_DEVICES=${GPU:-0}

DQN=model/lc_dqn_vs_random_dense_dd5/final.pt
BC=model/lc_ppo_dqninit_${TAG}/pretrained.pkt
mkdir -p "model/lc_ppo_dqninit_${TAG}"

echo "[dqninit:$TAG] gpu=$CUDA_VISIBLE_DEVICES start $(date '+%F %T')"

# ---- Stage 1: behavioral cloning DQN -> PPO ----
echo "[dqninit:$TAG] Stage 1 BC from $DQN"
python -u -m game.lc.imitate_dqn \
  --dqn-path "$DQN" --dqn-hidden-dim 512 \
  --no-lstm --hidden-dim 512 \
  --output "$BC" \
  --num-games 5000 --epochs 50 --batch-size 2048 \
  --max-discard-draws 5 || { echo "BC failed"; exit 1; }

# ---- Stage 2: PPO self-play finetune from the BC checkpoint ----
echo "[dqninit:$TAG] Stage 2 PPO finetune lr=$LR"
python -u -m game.lc.train \
  --pretrained "$BC" \
  --no-lstm --hidden-dim 512 --ent-coef 0.01 \
  --score-diff-reward --opponent-mode self_play \
  --max-discard-draws 5 \
  --critic-warmup-steps 0 \
  --learning-rate "$LR" --anneal-lr --target-kl None \
  --gae-lambda 0.95 --gamma 1 --opponent-sync-interval 50 \
  --num-envs 128 --num-steps 32 \
  --num-minibatches 4 --update-epochs 4 --clip-coef 0.2 \
  --total-timesteps 50000000 --wandb-project-name ppo-eco --seed 1 --track \
  --model-dir "model/lc_ppo_dqninit_${TAG}" \
  --exp-name "lc_ppo_dqninit_${TAG}"
echo "[dqninit:$TAG] done $(date '+%F %T') exit=$?"
