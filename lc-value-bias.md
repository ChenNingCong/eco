# Lost Cities: Value Function Bias Problem

## Observed Behavior
- Agent opens all 5 lanes and dumps cards indiscriminately
- Never draws from discard piles
- Very few investment cards played strategically
- 50M training doesn't fix the issue — it's not a convergence problem

## Root Cause: On-Policy Value Bias

The value function is estimated under the current stochastic policy (which includes entropy-driven exploration). This creates a feedback loop:

1. High entropy forces random card plays → cards get dumped into bad expeditions
2. Value function learns "opening a lane leads to bad outcomes" (because entropy *will* cause bad follow-up plays)
3. Policy update uses this biased value → avoids strategic lane opening
4. As entropy decays, policy is already stuck in "dump everything" local optimum
5. Value function never sees what disciplined play looks like

This is exactly the **cliff walking** problem from Sutton & Barto: on-policy methods learn to avoid the cliff *plus a safety margin* because the policy itself might walk off the edge. In Lost Cities, every open lane is a cliff (potential -20 × multiplier).

## Why It's Worse in Lost Cities Than TTR

- TTR: claiming routes has only upside (always positive points)
- Lost Cities: opening an expedition is a -20 commitment that only pays off with disciplined follow-up
- The penalty × multiplier structure amplifies losses from random play
- Investment cards multiply both gains AND the -20 penalty, making them especially toxic under entropy

## Potential Mitigations

### 1. Reduce Bootstrap Bias (GAE lambda / num_steps)
- Current: `gae_lambda=0.85`, `num_steps=64`, games last ~72 steps
- At lambda=0.85, advantage estimates rely heavily on bootstrapped value (which is biased)
- **Try `gae_lambda=1.0`**: pure Monte Carlo returns, no bootstrap bias
- **Try `num_steps=128`**: guarantees full episodes in each rollout, so terminal reward is always observed directly
- Pro: directly addresses the bias. Con: higher variance (MC tradeoff)

### 2. Lower Entropy from the Start
- Current: 0.1 → 0.01 exponential over 1M steps
- **Try `ent_coef=0.001` fixed**: minimal forced exploration
- Exploration still happens via large legal action space + self-play diversity
- Pro: value function sees near-deterministic play. Con: may under-explore

### 3. Split Action Space (Two-Phase Turn)
- Current: single action = card × play_type × draw_source (600 actions)
- Split into: (1) play phase (card + expedition/discard), (2) draw phase (deck/discard pile)
- Apply entropy only to draw phase (safe exploration) not play phase (dangerous)
- Pro: targeted exploration. Con: requires engine/framework changes

### 4. Off-Policy Methods
- DQN/SAC can decouple exploration policy from value estimation
- Value is estimated under the *optimal* policy, not the exploratory one
- Pro: clean fix. Con: major framework change, sample efficiency tradeoffs

### 5. Reward Shaping (Already Testing)
- Reduce cliff height: `new_color_penalty` = 18, 15, 10
- Results so far: -15 and -10 too aggressive (agent plays too many investments)
- -18 still running
- Pro: simple. Con: changes the game, doesn't fix the underlying bias

### 6. Curriculum / Staged Training  
- Phase 1: train with very low entropy (0.001) to learn basic strategy
- Phase 2: fine-tune with slightly higher entropy for robustness
- Pro: lets value function learn from disciplined play first. Con: two-stage complexity

## Most Promising Next Experiments
- **A**: `gae_lambda=1.0, num_steps=128` (pure MC, full episodes)
- **B**: `ent_coef=0.001` fixed (minimal entropy)
- **C**: Combine A+B: MC returns + low entropy
