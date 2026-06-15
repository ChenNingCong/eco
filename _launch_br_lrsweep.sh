#!/bin/bash
# DQN best-response LR SWEEP vs the frozen 50M PPO (exploitability test).
# Fixed reward = dense own-score (build big -> the user's "building beats it" hypothesis);
# sweeps the DQN learning rate to find which best/fastest finds an exploit.
# 'vs FROZEN-PPO' win-rate = exploitability. ~70-80% => 50M is exploitable; ~50-55% => robust.
set -u
source ~/programs/anaconda3/etc/profile.d/conda.sh && conda activate minecraft
cd /home/zzhang18/nchen3/eco
export CUDA_VISIBLE_DEVICES=${GPU:-0}
FROZEN=model/lc_ppo_dqninit_final50m/latest.pkt
COMMON="--frozen-ppo-path $FROZEN --frozen-ppo-hidden 512 \
 --dense-reward --raw-score-reward --max-discard-draws 5 --hidden-dim 512 \
 --num-envs 128 --total-timesteps 15000000 --track --wandb-project-name ppo-eco --seed 1"
run () { lr=$1; tag=$2; echo "[lr-sweep] === lr=$lr start $(date '+%F %T') ==="
  python -u -m game.lc.train_dqn $COMMON --learning-rate "$lr" \
    --model-dir "model/lc_br_lr$tag" --exp-name "lc_br_lr$tag"
  echo "[lr-sweep] === lr=$lr done $(date '+%F %T') ==="; }
run 1e-4   1e4
run 2.5e-4 25e4
run 5e-4   5e4
echo "[lr-sweep] ALL DONE $(date '+%F %T')"
