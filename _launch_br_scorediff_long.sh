#!/bin/bash
# CORRECTED exploitability test: DQN best-response vs the frozen 50M PPO,
# optimizing the TRUE game objective (score-diff / zero-sum), long budget.
#
# Why: a genuine best-response in a SYMMETRIC zero-sum game must clear 50%
# (copying the opponent already gives 50%; BR is >= that). The earlier builder
# (own-score) BR plateaued at 43% -- BELOW the copy floor -> it was a FAILED/
# misaligned exploiter, NOT proof of robustness. This run optimizes score-diff.
#
# DIAGNOSTIC on benchmark/vs_frozen/win_rate:
#   >> 50%  : the 50M IS exploitable.
#   ~= 50%  : near-Nash / robust.
#   <  50%  : our exploiter is too weak (memoryless DQN cannot represent a copy
#             of the stochastic PPO) -> test inconclusive, need history/memory.
set -u
source ~/programs/anaconda3/etc/profile.d/conda.sh && conda activate minecraft
cd /home/zzhang18/nchen3/eco
export CUDA_VISIBLE_DEVICES=${GPU:-0}
FROZEN=model/lc_ppo_dqninit_final50m/latest.pkt
python -u -m game.lc.train_dqn \
  --frozen-ppo-path "$FROZEN" --frozen-ppo-hidden 512 \
  --score-diff-reward --max-discard-draws 5 --hidden-dim 512 \
  --learning-rate 2.5e-4 --num-envs 128 --total-timesteps 100000000 \
  --track --wandb-project-name ppo-eco --seed 1 \
  --model-dir model/lc_br_scorediff_long --exp-name lc_br_scorediff_long
echo "[br-scorediff-long] DONE $(date '+%F %T')"
