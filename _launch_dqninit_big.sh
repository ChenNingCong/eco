#!/bin/bash
# CAPACITY TEST: same BC-init -> PPO self-play recipe as the h512 50M run, but with a much
# LARGER + DEEPER network (hidden 1024, mlp_depth 4 = 16.7M params vs 2.31M). Apples-to-apples
# (same teacher, reward, opponent, LR schedule, 50M budget) so any gain is attributable to
# capacity, not the reward. If vs-heuristic doesn't beat the h512 baseline (~73% settled),
# capacity isn't the bottleneck.
#
# Usage:  GPU=N bash _launch_dqninit_big.sh
set -u
source ~/programs/anaconda3/etc/profile.d/conda.sh && conda activate minecraft
cd /home/zzhang18/nchen3/eco
export CUDA_VISIBLE_DEVICES=${GPU:-0}
H=1024; D=4
DQN=model/lc_dqn_vs_random_dense_dd5/final.pt
BC=model/lc_ppo_dqninit_big/pretrained.pkt
mkdir -p model/lc_ppo_dqninit_big
echo "[big] BC (h$H d$D) from $DQN  $(date '+%F %T')"
python -u -m game.lc.imitate_dqn \
  --dqn-path "$DQN" --dqn-hidden-dim 512 \
  --no-lstm --hidden-dim $H --mlp-depth $D \
  --output "$BC" \
  --num-games 5000 --epochs 50 --batch-size 2048 --max-discard-draws 5 || { echo "BC failed"; exit 1; }

echo "[big] PPO finetune (h$H d$D)  $(date '+%F %T')"
python -u -m game.lc.train \
  --pretrained "$BC" \
  --no-lstm --hidden-dim $H --mlp-depth $D --ent-coef 0.01 \
  --score-diff-reward --opponent-mode self_play \
  --max-discard-draws 5 --critic-warmup-steps 0 \
  --learning-rate 2.5e-4 --anneal-lr --target-kl None \
  --gae-lambda 0.95 --gamma 1 --opponent-sync-interval 50 \
  --num-envs 128 --num-steps 32 --num-minibatches 4 --update-epochs 4 --clip-coef 0.2 \
  --save-interval 2000000 \
  --total-timesteps 50000000 --wandb-project-name ppo-eco --seed 1 --track \
  --model-dir model/lc_ppo_dqninit_big --exp-name lc_ppo_dqninit_big
echo "[big] done $(date '+%F %T') exit=$?"
