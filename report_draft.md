# R-oko PPO Training: Ablation Study

## Project Overview

R-oko is a **multiplayer card game** (2-5 players) where players collect recycling cards and play them to claim factory stack rewards worth varying point values. Players must manage hand limits (max 5 cards), decide which color to specialise in, and time their factory card claims. Excess cards become penalties (-1 point each), while having *zero* penalties earns a bonus. The game ends when any factory stack is emptied, and the player with the highest score wins.

We train a PPO agent via **3-player self-play**. While the game supports 2-5 players, 3-player is the most strategically interesting: it introduces multi-agent dynamics (non-transitive strategies, coalition effects) that don't exist in 2-player games, without the computational cost of 4-5 player training.

**Try it yourself:** Play against the trained agent at [huggingface.co/spaces/NingcongChen/r-oko](https://huggingface.co/spaces/NingcongChen/r-oko). Full source code is available at [github.com/ChenNingCong/eco](https://github.com/ChenNingCong/eco).

## Training Approach

### PPO Self-Play (No Search)

Most well-known game-playing agents (AlphaGo, AlphaZero, MuZero) combine neural networks with **Monte Carlo Tree Search (MCTS)** -- using the network to guide a look-ahead search at each decision point. While powerful, MCTS adds significant complexity: it requires a forward model of the environment, is expensive at inference time, and is tricky to parallelise efficiently.

We take a simpler approach: **pure PPO (Proximal Policy Optimization) with self-play**, with no search at all. The agent plays against copies of itself, all sharing the same policy weights, and learns a direct mapping from board state to action probabilities through trial-and-error. This is feasible for R-oko because the game is relatively short (~15-30 actions per player per game) and the action space is modest (108 discrete actions, heavily masked by game rules).

**How self-play works in practice:**
1. Each episode, the agent is **randomly assigned** to one of the 3 seats, ensuring symmetric training across all positions.
2. All 128 parallel environments run simultaneously. In each environment, one seat is the "learning agent" whose transitions are collected for PPO updates.
3. The remaining seats are **opponents driven by the same policy network**. Opponent inference is batched across all environments using a generator-based coroutine architecture: instead of N separate forward passes per opponent turn, all pending opponent observations are stacked and evaluated in a single batched call.
4. After each rollout (32 steps x 128 envs = 4096 transitions), advantages are computed via GAE, then the policy and value networks are updated via PPO's clipped objective over 4 epochs of 4 minibatches.
5. Periodically, the agent is benchmarked against a random opponent to track progress.

**Why self-play for multiplayer games:** In 2-player zero-sum games, self-play provably converges to a Nash equilibrium (minimax theorem). In 3+ player games, this guarantee doesn't hold -- Nash equilibria are not unique, and strategy cycling can occur (agent A beats B beats C beats A). Despite this theoretical limitation, self-play remains practical: the agent learns robust strategies by continuously adapting to its own improving play. We observe that the agent's win rate against a fixed random opponent increases monotonically, even though cross-play between checkpoints shows approximate parity (as expected in a near-symmetric game).

**Terminal reward**: +1 for winning, -1 for losing (based on final scores). **No intermediate reward shaping** is used in the ablation runs -- the agent must learn purely from game outcomes. This makes value learning challenging: the critic must predict win/loss from the current board state, receiving learning signal only at the end of each game (~15-30 actions per episode).

**Training speed:** With 128 parallel environments on a single GPU, the full 50M-sample training run completes in **under 6 hours**. The 10M ablation runs each finish in about 1 hour. This fast iteration cycle allowed us to run multiple controlled experiments and identify which techniques actually matter.

### Network Architecture and Action Masking

The network is a **separate actor-critic** with a shared observation encoder:
- **Shared encoder**: all float features -> 2-layer MLP (256 hidden, LayerNorm + ReLU)
- **Actor trunk**: 2-layer MLP (256 hidden) -> 108-dim action logits
- **Critic trunk**: 2-layer MLP (256 hidden) -> scalar value

R-oko has 108 discrete actions, but at any given state only a small subset is legal -- typically 5-15 actions depending on the hand and game phase. **Action masking** is critical for making training tractable:

- Illegal actions receive logit = -1e8 before softmax, making them effectively impossible to sample
- Without masking, the agent would waste the vast majority of its experience on illegal moves that produce no learning signal (the environment rejects them)
- Masking also prevents catastrophic game-logic errors -- in a complex card game, many action combinations are invalid (e.g., playing cards you don't have, discarding when not over hand limit)
- A hard mask enforcement safety net catches any remaining edge cases: if sampling still produces an illegal action (numerically possible with -1e8 rather than -inf), it falls back to argmax over legal actions

Action masking effectively reduces the exploration problem from 108 actions to ~10, making PPO viable for this game without any search.

### Adaptive Learning Rate via Target KL (Rudin et al., 2022)

We adopt the adaptive learning rate mechanism from **"Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning"** (Rudin et al., 2022). Instead of a fixed learning rate or simple annealing, we set a **target KL divergence** (`target_kl = 0.01`) between the old and new policy:

- After each PPO update epoch, we measure the approximate KL divergence
- If `KL > 2 * target_kl`, the learning rate is decreased by 1.5x
- If `KL < 0.5 * target_kl`, it is increased by 1.5x (clamped to [1e-5, 1e-2])
- If `KL > target_kl` during any of the 4 update epochs, we **break early** -- preventing over-optimisation on stale data

This acts as an automatic step-size controller: it prevents destructively large policy updates that cause performance collapse, while allowing faster learning when the policy is changing slowly. Combined with gradient clipping (`max_grad_norm = 0.5`) and advantage normalisation, this provides robust training across different phases without manual LR scheduling.

## Game Representation

### Modelling as a Perfect Information Game

The physical R-oko game is **imperfect information**: players' hands are private (you can't see what cards opponents hold), and the draw pile order is unknown. These two sources of hidden information normally require belief tracking -- reasoning about what opponents *might* have based on their actions.

We deliberately simplify this by modelling the game as **perfect information**: all players' hands are fully visible in the observation, and the draw pile composition (count of each color x type remaining) is included as 8 normalised floats. The agent sees everything except the draw pile *order*.

This is a conscious design choice with clear trade-offs:
- **Upside**: eliminates the need for belief tracking or memory (no LSTM/Transformer required), letting a simple feedforward MLP focus purely on strategic planning under known state
- **Downside**: the agent learns to play a different (easier) game than the physical one -- it can see opponents' hands and plan accordingly, which real players cannot do
- **Justification**: in practice, experienced R-oko players can infer much of this information -- hands change visibly as cards are played and picked up, and card counting reveals the draw pile composition. The perfect-information version is a reasonable upper bound on what a strong player could deduce.

### Observation Space

The observation is a structured vector (**156 floats** for 3 players) with two embedded discrete tokens. Each field is included for a specific reason:

**Discrete tokens** (embedded via learned 32-dim embeddings):
- **phase** (0=play, 1=discard): The game has two distinct action phases with completely different action subsets. During the *play phase*, the agent chooses which cards to play on a recycling factory (100 possible actions encoding color x singles x doubles); during the *discard phase*, it must discard excess cards when the hand limit is exceeded (8 possible actions encoding color x type). The phase token tells the network which decision type it's facing, complementing the action mask which enforces legality.
- **current_player**: Always 0 under relative seat encoding (see below).

**Continuous fields** (all normalised to roughly [0, 1]):

| Field | Dim | Why it's needed |
|-------|-----|-----------------|
| **hands** | 24 | All players' card holdings (count per color x type). Essential for predicting opponents' likely plays and planning which colors to contest. |
| **recycling_side** | 8 | Cards currently on the recycling side of each factory. Determines whether playing more cards this turn will reach the threshold (total value >= 4) to claim a factory card. |
| **waste_side** | 32 | Cards on the waste side of each factory (color x color x type). The active player *must* pick up all waste cards of the color they play -- so this directly affects the hand-bloat cost of playing a given color. Strategically important: dumping waste onto a factory that your opponent is about to play forces them to pick it up. |
| **factory_stacks** | 32 | Remaining factory card values per color (consumed slots = -1, active slots normalised by 5). This is the reward landscape: the agent needs to know what the next claimable factory card is worth (+5 is great, -2 is a penalty), and which stacks are nearly empty (triggering game end). |
| **collected** | 24 | Per-player collected factory cards: a scoring flag (count > 1) and value sum per color. Critical for the **winning condition**: a player must have *more than one* factory card of a color for it to score anything. A single card of a color scores zero points. This encoding lets the agent reason about whether to invest in a color (needs 2+ cards) or abandon it. |
| **penalty_pile** | 24 | Per-player penalty card composition. Penalty cards cost -1 point each, but having *zero* penalties earns a bonus equal to the number of opponents who do have penalties (+2 in 3-player). This creates a strong incentive to avoid penalties while hoping opponents accumulate them -- the agent needs to track both its own and opponents' penalty status. |
| **scores** | 3 | Current normalised scores for all players. Lets the value function assess relative standing. |
| **draw_pile_size** | 1 | How many cards remain in the deck. Signals proximity to reshuffling and indirectly to game end. |
| **draw_pile_comp** | 8 | Remaining cards by (color, type), normalised by max count. The key "perfect information" feature: lets the agent reason about the probability distribution over future draws. Without this, the value function struggles to predict outcomes -- this was confirmed empirically when adding draw_pile_comp significantly improved explained variance. |

### Relative Seat Encoding

All player-indexed observations are **rotated** so the agent always sees itself at index 0. "My hand" is always the first slot in the hands array, "my score" is always scores[0], "my collected cards" are always collected[0:8], etc. The current_player token is fixed at 0.

Both absolute encoding (player index 0/1/2) and relative encoding are valid approaches. We use relative encoding throughout because it:
- Halves the effective problem: the network sees one canonical perspective instead of three
- Removes the need to learn that "player 1's hand when I'm player 1" means the same as "player 0's hand when I'm player 0"
- Makes the agent seat-agnostic by construction, which is important for self-play where the agent is randomly assigned to any seat each episode

We have not rigorously ablated relative vs absolute encoding (the absolute_seat run crashed early due to an unrelated bug), but relative encoding is standard practice in multi-agent RL and we adopt it as a default.

### Benchmarking Against a Random Agent

During training, we periodically evaluate the agent against a **random opponent** that selects uniformly among legal actions. While this is a weak baseline, it serves as a reliable proxy for monitoring training progress:

- **Monotonic signal**: unlike self-play metrics (which can oscillate due to co-adaptation), win rate against a fixed random opponent increases monotonically as the agent improves. This provides an unambiguous measure of whether training is making progress.
- **Cheap to compute**: each evaluation runs 200 games with no gradient computation, adding negligible overhead to the training loop.
- **Sensitive in the early-to-mid training regime**: a random agent in 3-player R-oko wins ~33% of games by chance. An agent that has learned basic strategy quickly climbs to 70-80%, and the gap from 80% to 99% reflects increasingly sophisticated play (hand management, factory timing, opponent awareness). The metric only saturates once the agent is near-optimal.
- **Mean score as a secondary metric**: when win rate saturates near 99%, the agent's mean score against random opponents continues to differentiate quality — a stronger agent wins by larger margins.

In a tournament setting between trained checkpoints, win rates hover near 33% (as expected for agents of similar strength in a symmetric 3-player game), making cross-play uninformative for tracking improvement. The random baseline avoids this problem by providing a fixed reference point.

## 1. Ablation Study: 10M Samples

We isolate the effect of two training techniques through four controlled experiments, each trained for **10M environment samples** with identical hyperparameters (adaptive LR via target_kl=0.01, 128 envs, 32 steps/rollout, 4 minibatches, 4 epochs, gamma=1.0).

### A Note on Explained Variance

Before diving into results, it's worth understanding **explained variance** -- the key diagnostic metric in this study. It measures how well the critic's value predictions match actual returns: `1 - Var(returns - predictions) / Var(returns)`. A value of 1.0 means perfect prediction; 0.0 means no better than predicting the mean.

This metric is *especially* important here because we use **no reward shaping** -- the only signal is terminal +1/-1. In this setting:
- The critic must learn to predict win probability purely from board state
- Advantage estimates (used for policy updates) depend entirely on the critic's accuracy
- If the critic is poor (low explained variance), the advantage signal is dominated by noise, and training stalls

In games with dense intermediate rewards, a poor critic is partially compensated by the immediate reward. Here, there are no intermediate rewards: the critic's value estimate *is* the learning signal. Explained variance directly measures whether training can make progress.

### Results

| Condition | Entropy Coef | GAE Lambda | Win Rate vs Random | Expl. Variance | Value Loss |
|-----------|-------------|------------|-------------------|----------------|------------|
| Baseline | 0 (none) | 1.0 | ~89% | ~0.16 | ~0.29 |
| Entropy only | 0.1 -> 0.01 | 1.0 | ~93% | ~0.20 | ~0.26 |
| GAE only | 0 (none) | 0.85 | ~93% | ~0.39 | ~0.07 |
| **Both** | **0.1 -> 0.01** | **0.85** | **~99%** | **~0.45** | **~0.05** |

*(Metrics are averaged over the last 10 logged data points to reduce noise.)*

### Entropy Annealing (0.1 -> 0.01)

Without entropy regularisation, the policy collapses early: the agent finds a reasonable strategy and stops exploring. The entropy of the non-annealed runs drops to 0.06-0.09 (nearly deterministic), meaning the agent commits to a fixed strategy regardless of board state.

**Entropy annealing** applies a linearly decaying entropy bonus: starting at coefficient 0.1 (strong exploration pressure) and decaying to 0.01 over 40k update steps. Early in training, this prevents the policy from collapsing before the critic has learned useful value estimates. Later, the reduced bonus allows the policy to sharpen.

**Impact**: +4% win rate over baseline (89% -> 93%). The annealed runs maintain entropy of 0.10-0.32 throughout training, indicating the policy retains meaningful stochasticity and adapts its play to different board states rather than following a single fixed strategy.

### GAE Lambda = 0.85

With gamma=1.0 (no discounting, appropriate for episodic games), GAE with lambda=1.0 reduces to **Monte Carlo returns** -- the advantage is computed from the full trajectory return. This is unbiased but has high variance, because a single game outcome (+1/-1) must be attributed across all ~20 actions in the episode.

Reducing lambda to 0.85 introduces the critic's value estimates into the advantage calculation, trading slight bias for dramatically lower variance. The critic doesn't need to be perfect -- even a rough value estimate reduces the noise in the policy gradient.

**Impact**: +4% win rate (89% -> 93%), but the real story is value learning. GAE=0.85 achieves **~0.39 explained variance** vs ~0.16 for the baseline -- the critic learns a meaningfully useful value function. Value loss drops from ~0.29 to ~0.07 (4x improvement). This creates a positive feedback loop: better value estimates -> lower-variance advantages -> more stable policy updates -> the critic can learn from more consistent behaviour.

### Combined Effect: Synergy, Not Addition

The two techniques are **synergistic**: combined they achieve ~99% win rate, far exceeding the ~96% you'd expect from adding their individual improvements. Why?

- **Entropy without GAE**: The agent explores diverse strategies, but the high-variance critic can't tell which strategies are actually better. Exploration is partially wasted.
- **GAE without entropy**: The critic learns accurate values, but the policy collapses before the critic has converged. The critic ends up evaluating a narrow, suboptimal strategy.
- **Both**: Entropy keeps the policy exploring long enough for the GAE-improved critic to converge. The accurate critic then guides exploration toward genuinely better strategies. Each technique compensates for the other's weakness.

## 2. Training Scale: 10M vs 50M

Using the best configuration (entropy annealing + GAE=0.85), we train for **50M samples** to understand scaling behaviour.

| Metric | 10M | 50M |
|--------|-----|-----|
| Win Rate vs Random | ~99% | ~99% |
| Explained Variance | ~0.45 | ~0.43 |
| Value Loss | ~0.05 | ~0.07 |
| Entropy | ~0.32 | ~0.20 |

*(Metrics averaged over last 10 data points.)*

### The Agent Keeps Improving Beyond 10M

The win rate vs random saturates near 99% and can no longer differentiate agent quality at this level. But the **training curves** tell a richer story:

- The win rate **climbs steadily through the first 20M samples** before plateauing. The agent is learning meaningful strategy improvements well past 10M.
- Policy entropy decreases from ~0.32 to ~0.20, indicating the agent is becoming more **decisive** -- committing more strongly to its preferred actions as it becomes more confident in its strategy.
- Value loss and explained variance stabilise, suggesting the critic has converged to the best value function achievable given the observation space and network capacity.

### Saturation Around 30M Samples

All metrics flatten after approximately 30M samples. The agent has extracted most of the learnable signal from self-play at this difficulty level.

## Summary

Two simple techniques transform a stagnating baseline (~89% win rate, ~0.16 explained variance) into a near-perfect agent (~99% win rate, ~0.45 explained variance):

1. **Entropy annealing** (0.1 -> 0.01) prevents premature policy collapse, keeping the agent exploring until the critic can guide it effectively
2. **GAE lambda = 0.85** enables accurate value learning despite terminal-only rewards, breaking the high-variance bottleneck of Monte Carlo returns

Neither technique alone is sufficient -- their **combination is critical** and produces a synergistic effect far exceeding the sum of their individual improvements.

Combined with **adaptive learning rate via target KL** (from Rudin et al., 2022) for stable training, the agent reaches competence within 10M samples and saturates by ~30M, suggesting the next frontier is stronger training opponents rather than longer training.
