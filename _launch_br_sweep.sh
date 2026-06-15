#!/bin/bash
# EXPLOITABILITY SWEEP: train DQN best-responses against the frozen 50M PPO agent.
# Each config trains vs the SAME frozen opponent (model/lc_ppo_dqninit_final50m); the
# 'vs FROZEN-PPO' benchmark = how exploitable the 50M is. If any config's win-rate climbs
# to ~70-80%, the 50M is exploitable (the user is right it's weak). If all plateau ~50-55%,
# it's robust/near-unexploitable. Sweeps the REWARD (what kind of exploit):
#   dense_raw = pure own-score (build big -> tests "building beats it")
#   dense_win = own-score + terminal win bonus (build AND win)
#   scorediff = zero-sum (directly maximize beating the 50M)
# Runs sequentially. ~15M steps each (DQN converges fast).
set -u
source ~/programs/anaconda3/etc/profile.d/conda.sh && conda activate minecraft
cd /home/zzhang18/nchen3/eco
export CUDA_VISIBLE_DEVICES=${GPU:-0}
FROZEN=model/lc_ppo_dqninit_final50m/latest.pkt
COMMON="--frozen-ppo-path $FROZEN --frozen-ppo-hidden 512 --max-discard-draws 5 --hidden-dim 512 \
 --learning-rate 1e-4 --num-envs 128 --total-timesteps 15000000 \
 --track --wandb-project-name ppo-eco --seed 1"
run () { tag=$1; shift; echo "[br-sweep] === $tag start $(date '+%F %T') ==="
  python -u -m game.lc.train_dqn $COMMON "$@" --model-dir "model/lc_br_$tag" --exp-name "lc_br_$tag"
  echo "[br-sweep] === $tag done $(date '+%F %T') ==="; }
run dense_raw --dense-reward --raw-score-reward
run dense_win --dense-reward
run scorediff --score-diff-reward
echo "[br-sweep] ALL DONE $(date '+%F %T')"
