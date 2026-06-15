# BC→PPO (DQN-init) pipeline for Lost Cities — a working recipe

**Key finding.** From-scratch self-play PPO **collapses** to a lazy degenerate policy:
it drives the probability of ever playing an investment card to **exactly 0** (an
*absorbing* state for policy gradients — never sampled ⇒ never reinforced ⇒ stays 0),
opens all 5 lanes, never invests, and **loses to the heuristic (~35%)**. See the
"investment barrier" diagnosis: investing is *get-worse-before-better*, so the on-policy
gradient pushes away from it, and a deterministic DQN can cross that barrier (bootstrapping)
but from-scratch PPO cannot.

**The fix that works: behavior-clone a strong DQN builder, then PPO-finetune.**
This seeds PPO *past* the barrier (already investing), so PPO refines instead of having to
discover-across-a-barrier it can't cross. The result (`lc_ppo_dqninit_ent01`) **improves
continuously** and is still climbing at 29M+ steps:

| training third (vs heuristic, dd5-capped eval, mean of ~125 benchmarks each) | win | score | inv |
|---|---|---|---|
| first  | 49.3 ± 0.8% | 15.5 | 2.69 |
| middle | 57.8 ± 0.5% | 21.7 | 2.52 |
| last   | **62.5 ± 0.5%** | **26.7** | 2.68 |

**Why it climbs *slowly*:** Lost Cities is high-variance (the deck shuffle dominates close
games), and PPO is **on-policy** (each gradient step only learns from freshly-sampled
trajectories), so refinement past a competent builder is sample-inefficient. The agent keeps
getting better — it just needs many steps to accumulate signal through the noise. (Measure with
many benchmarks averaged, not single 100-game snapshots — per-benchmark SEM is large.)

> NOTE on naming: the strong teacher `dense_dd5` is trained with `--dense-reward --raw-score-reward`
> = **pure own-score** (general-sum) + dd5 cap; it beats the heuristic ~65%, invests ~3.2. The
> zero-sum `score-diff` DQN is *weaker* (loses to heuristic, 32%). LC is a weak-interaction game,
> so own-score is a clean, strong training signal.

## Pipeline (reproduce with `_launch_dqninit_ppo.sh`)

### Stage 1 — Behavior cloning (DQN → PPO), `game/lc/imitate_dqn.py`
Clone the **strong own-score builder** DQN's greedy policy into a no-LSTM PPO agent.
```
python -u -m game.lc.imitate_dqn \
  --dqn-path model/lc_dqn_vs_random_dense_dd5/final.pt --dqn-hidden-dim 512 \
  --no-lstm --hidden-dim 512 \
  --output model/lc_ppo_dqninit_ent01/pretrained.pkt \
  --num-games 5000 --epochs 50 --batch-size 2048 --max-discard-draws 5
```
Result: the clone already invests (inv ~3.1) and beats the heuristic ~54% — i.e. it starts
*in the strong basin*, not the lazy collapse. (BC is lossy, ~0.6 action-match, hence < the
teacher's 65% — recovering/exceeding that is the finetune's job.)

### Stage 2 — PPO self-play finetune, `game/lc/train.py`
```
python -u -m game.lc.train \
  --pretrained model/lc_ppo_dqninit_ent01/pretrained.pkt \
  --no-lstm --hidden-dim 512 --ent-coef 0.01 \
  --score-diff-reward --opponent-mode self_play \
  --max-discard-draws 5 --critic-warmup-steps 0 \
  --learning-rate 2.5e-4 --anneal-lr --target-kl None \
  --gae-lambda 0.95 --gamma 1 --opponent-sync-interval 50 \
  --num-envs 128 --num-steps 32 --num-minibatches 4 --update-epochs 4 --clip-coef 0.2 \
  --total-timesteps 50000000 --seed 1 --track \
  --model-dir model/lc_ppo_dqninit_ent01 --exp-name lc_ppo_dqninit_ent01
```

**Config rationale:**
- **TERMINAL `score-diff`** (zero-sum (own−opp)/30 at game end), NOT dense. Terminal so the
  per-step investment barrier doesn't erode the BC-seeded investing; zero-sum so the objective
  is competitive.
- **dd5 discard cap** (`--max-discard-draws 5`) — blocks the discard-pile storage hack and
  matches the teacher's training.
- **Frozen self-play** opponent (`--opponent-sync-interval 50`) — more stable than a live mirror.
- **`--ent-coef 0.01`**, fixed LR `2.5e-4` annealed (`--anneal-lr --target-kl None`), `gae 0.95`,
  `gamma 1`, feedforward (`--no-lstm`), hidden 512.
- `--critic-warmup-steps 0` because the critic is already BC-pretrained.

**Eval note / bug fix:** the benchmark factory must use the *same* discard cap as training —
`LCTrainer._bench_factory` now passes `max_discard_draws=config.max_discard_draws` (it previously
defaulted to uncapped, so every PPO benchmark was silently running uncapped discard, ddraw ~25).
