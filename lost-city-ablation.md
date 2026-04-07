# Lost Cities — Decomposed Action Ablation Results

## Action Space Variants
- **Flat**: 600 actions = card_id(50) × action_type(2) × draw_source(6)
- **Decomposed (2-phase)**: Play phase: card_id(50) × action_type(2) = 100 actions, Draw phase: 6 actions → 106 total
- **Decomposed (3-phase)**: Select card(50) + action type(2) + draw source(6) = 58 total
- **Product (factored heads)**: 3 heads — card(50), type(2), draw(6) — combined via additive log-space outer sum to produce 600 logits. Rank-1 approximation of the full logit tensor. Structured but lossy.

## Two Failure Modes

### 1. Discard-draw (ddraw) explosion
- **Trigger**: MC-like returns (high GAE lambda, high n_steps, large num_envs like 2048) + LSTM
- **Symptom**: ddraw metric grows to 100–1000+, agent stuck in degenerate discard-draw cycles
- **Root cause**: LSTM accumulates state that, combined with high-variance MC-like returns, exploits degenerate discard-draw patterns
- **Does NOT happen with**: FF networks (no memory), or small 128-env configs with 64 n_steps + gae=0.85 (sufficiently TD-like)

### 2. Conservative low-lane play
- **Trigger A**: gamma=0.999 + [0,1] reward → agent opens only 3–4 lanes, negative scores
- **Trigger B**: blind_draw (no played card obs in draw phase) + LSTM → collapses to 1 lane, score ~0
- **Root cause**: Agent lacks information or incentive to take risks → retreats to safe strategy (fewer lanes avoids -20 penalty per lane)

## All Runs

| Exp Name | State | Envs | nstep | GAE | Gamma | Arch | Reward | Action | H | Score |
|---|---|---|---|---|---|---|---|---|---|---|
| lc_p15 | finished | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | flat | 256 | 55 |
| lc_p10 | finished | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | flat | 256 | 45 |
| lc_baseline | finished | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | flat | 256 | 40 |
| lc_baseline_50m | finished | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | flat | 256 | 39 |
| lc_p18_50m | finished | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | flat | 256 | 36 |
| lc_baseline_50m_ff | running | 128 | 64 | 0.85 | 1.0 | FF | +1/-1 | flat | 256 | 36 |
| lc_scorediff | finished | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | flat | 256 | 34 |
| lc_p18 | finished | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | flat | 256 | 33 |
| lc_maxlanes4 | finished | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | flat | 256 | 26 |
| lc_decompose_lanes5_env128 | running | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | 2phase | 256 | 25 |
| lc_decompose_lanes5_ff | running | 128 | 64 | 0.85 | 1.0 | FF | +1/-1 | 2phase | 256 | 22 |
| lc_decompose_lanes5_01reward_ff | running | 2048 | 64 | 0.85 | 0.999 | FF | [0,1] | 2phase | 256 | -8 |
| lc_decompose_lanes5_01reward_ff_ent2m | running | 2048 | 64 | 0.85 | 0.999 | FF | [0,1] | 2phase | 256 | -10 |
| lc_decompose_lanes5 | crashed | 2048 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | 2phase | 256 | -23 |
| lc_decompose_lanes5_gae95 | crashed | 2048 | 64 | 0.95 | 1.0 | LSTM | +1/-1 | 2phase | 256 | -21 |
| lc_decompose_lanes5_nstep128 | killed | 128 | 128 | 0.85 | 1.0 | LSTM | +1/-1 | 2phase | 256 | — |
| lc_decompose_lanes5_blind | crashed | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | 2phase | 256 | -1 |
| lc_3phase_ff | running | 128 | 64 | 0.85 | 1.0 | FF | +1/-1 | 3phase | 256 | early |
| lc_3phase_01reward_ff | running | 2048 | 64 | 0.85 | 0.999 | FF | [0,1] | 3phase | 256 | early |
| lc_product_ff | running | 128 | 64 | 0.85 | 1.0 | FF | +1/-1 | product | 256 | early |
| lc_50m_h512 | running | 128 | 64 | 0.85 | 1.0 | LSTM | +1/-1 | flat | 512 | early |

## Key Conclusions

1. **Flat action space is more robust**: LSTM flat baseline scores ~39 vs decomposed ~25. Flat never triggers ddraw explosion.
2. **Decomposed is significantly worse than flat**: Even in the best decomposed config (128 envs, LSTM, score ~25), it trails the flat baseline (~39) by a large margin. The 2-phase decomposition (play 100 + draw 6) may lose information by splitting the joint play+draw decision, making credit assignment harder (play reward is delayed one extra step). The doubled number of steps per game also halves the complete games seen per rollout.
3. **Decomposed works at small scale**: 128 envs / 64 n_steps / gae=0.85 / gamma=1.0 / +1/-1 reward is a safe config for both LSTM and FF.
4. **LSTM + MC = ddraw explosion**: Large envs or long rollouts make returns MC-like, and LSTM exploits this to find degenerate discard-draw cycles. FF is immune. Only observed with decomposed actions — flat action space never triggers this.
5. **Weak signal = conservative play**: Removing negative reward (01 reward) or observation info (blind_draw) causes the agent to retreat to a low-lane, risk-averse strategy.
6. **LSTM > FF when stable**: LSTM decomposed (25) > FF decomposed (22), LSTM flat (39) > FF flat (36).
