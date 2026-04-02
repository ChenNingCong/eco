"""
Create a wandb report comparing training runs.

Usage: python create_report.py
"""

import wandb_workspaces.reports.v2 as wr

ENTITY = "ningcong-chen"
PROJECT = "ppo-eco"

# Run names
ABL_BASELINE = "Eco-v0__abl_baseline__1__1775138633"
ABL_ENTROPY  = "Eco-v0__abl_entropy__1__1775138633"
ABL_GAE      = "Eco-v0__abl_gae__1__1775138633"
ENT_GAE_10M  = "Eco-v0__ent_gae_10m__1__1775076276"
ENT_GAE_50M  = "Eco-v0__ent_gae_50m__1__1775092102"

FILTER_10M = f"Name in ['{ABL_BASELINE}', '{ABL_ENTROPY}', '{ABL_GAE}', '{ENT_GAE_10M}']"
FILTER_SCALE = f"Name in ['{ENT_GAE_10M}', '{ENT_GAE_50M}']"

M = wr.MarkdownBlock


def make_panel(metric, title):
    return wr.LinePlot(x="global_step", y=[metric], title=title, smoothing_factor=0.6)


def create_report():
    report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="R-oko PPO Training: Ablation Study",
        description="Training a PPO agent via 3-player self-play for the R-oko card game. "
                    "Ablation study isolating entropy annealing and GAE lambda, plus training scale analysis.",
        blocks=[
            # ══════════════════════════════════════════════════════════════
            # OVERVIEW
            # ══════════════════════════════════════════════════════════════
            wr.H1("Project Overview"),
            M(
                "R-öko is a **multiplayer card game** (2-5 players) where players collect recycling cards "
                "(88 total: 4 colors × 22 cards) and play them to claim factory stack rewards worth varying point values. "
                "Players must manage hand limits (max 5 cards), decide which colors to specialise in, and time their "
                "factory card claims. Excess cards become penalties (-1 point each), while having zero penalties earns "
                "a bonus. The game ends when any factory stack is emptied.\n\n"
                "We train a PPO agent via **3-player self-play**. While the game supports 2-5 players, "
                "3-player is the most strategically interesting: it introduces multi-agent dynamics "
                "(non-transitive strategies, coalition effects) that don't exist in 2-player games, "
                "without the computational cost of 4-5 player training.\n\n"
                "**Try it yourself:** Play against the trained agent at "
                "[huggingface.co/spaces/NingcongChen/r-oko](https://huggingface.co/spaces/NingcongChen/r-oko). "
                "Full source code: [github.com/ChenNingCong/eco](https://github.com/ChenNingCong/eco)."
            ),

            # ══════════════════════════════════════════════════════════════
            # TRAINING APPROACH
            # ══════════════════════════════════════════════════════════════
            wr.H1("Training Approach"),

            wr.H2("PPO Self-Play (No Search)"),
            M(
                "Most well-known game-playing agents (AlphaGo, AlphaZero, MuZero) combine neural networks with "
                "**Monte Carlo Tree Search (MCTS)** — using the network to guide a look-ahead search at each "
                "decision point. While powerful, MCTS adds significant complexity: it requires a forward model, "
                "is expensive at inference time, and is tricky to parallelise.\n\n"
                "We take a simpler approach: **pure PPO with self-play**, no search at all. The agent plays "
                "against copies of itself (all sharing the same policy weights) and learns a direct mapping from "
                "board state to action probabilities through trial-and-error. This is feasible because R-öko "
                "games are short (~15-30 actions per player) and the action space is modest (108 discrete "
                "actions, heavily masked by game rules).\n\n"
                "**How it works:**\n"
                "1. Each episode, the agent is **randomly assigned** to one of the 3 seats for symmetric training\n"
                "2. 128 parallel environments run simultaneously — one seat per env collects transitions for PPO updates\n"
                "3. Remaining seats are **opponents driven by the same policy**, with inference batched across all "
                "envs via a generator-based coroutine architecture (one batched forward pass per opponent round, not per env)\n"
                "4. After each rollout (32 steps × 128 envs = 4,096 transitions), advantages are computed via GAE, "
                "then the networks are updated via PPO's clipped objective (4 epochs, 4 minibatches)\n\n"
                "**Why self-play for multiplayer games:** In 2-player zero-sum games, self-play provably converges "
                "to a Nash equilibrium. In 3+ player games this guarantee doesn't hold — strategy cycling can "
                "occur. Despite this, self-play remains practical: the agent learns robust strategies by continuously "
                "adapting to its own improving play.\n\n"
                "**Terminal reward only:** +1 for winning, -1 for losing. No intermediate reward shaping — "
                "the agent must learn purely from game outcomes, making value learning especially challenging.\n\n"
                "**Training speed:** With 128 parallel environments on a single GPU, 50M samples completes in "
                "**under 6 hours**; each 10M ablation run finishes in ~1 hour."
            ),

            wr.H2("Network Architecture and Action Masking"),
            M(
                "**Separate actor-critic** with shared observation encoder:\n"
                "- **Shared encoder**: float features → 2-layer MLP (256 hidden, LayerNorm + ReLU)\n"
                "- **Actor trunk**: 2-layer MLP (256 hidden) → 108-dim action logits\n"
                "- **Critic trunk**: 2-layer MLP (256 hidden) → scalar value\n\n"
                "R-öko has 108 discrete actions, but only ~5-15 are legal at any state. "
                "**Action masking** is critical for making training tractable:\n"
                "- Illegal actions get logit = -1e8 before softmax, making them effectively impossible to sample\n"
                "- Without masking, the agent wastes experience on illegal moves with no learning signal\n"
                "- A hard mask enforcement safety net catches edge cases (falls back to argmax over legal actions)\n\n"
                "Action masking reduces the exploration problem from 108 actions to ~10, "
                "making PPO viable for this game without any search."
            ),

            wr.H2("Adaptive Learning Rate via Target KL (Rudin et al., 2022)"),
            M(
                "We adopt the adaptive LR mechanism from **\"Learning to Walk in Minutes Using Massively Parallel "
                "Deep Reinforcement Learning\"** (Rudin et al., 2022). Instead of fixed LR or annealing, "
                "we set a **target KL divergence** (`target_kl = 0.01`):\n\n"
                "- If `KL > 2× target` after an update epoch → LR decreased by 1.5×\n"
                "- If `KL < 0.5× target` → LR increased by 1.5× (clamped to [1e-5, 1e-2])\n"
                "- If `KL > target` during any of the 4 update epochs → **break early**\n\n"
                "This acts as an automatic step-size controller: prevents destructively large updates "
                "while allowing faster learning when the policy is stable. Combined with gradient clipping "
                "(`max_grad_norm = 0.5`) and advantage normalisation for robust training."
            ),

            # ══════════════════════════════════════════════════════════════
            # GAME REPRESENTATION
            # ══════════════════════════════════════════════════════════════
            wr.H1("Game Representation"),

            wr.H2("Perfect Information Modelling"),
            M(
                "The physical R-öko game is **imperfect information**: players' hands are private (you can't "
                "see what cards opponents hold), and the draw pile order is unknown. These two sources of hidden "
                "information normally require belief tracking — reasoning about what opponents *might* have "
                "based on their actions.\n\n"
                "We deliberately simplify this by modelling the game as **perfect information**: all players' "
                "hands are fully visible in the observation, and the draw pile composition (count of each "
                "color × type remaining) is included as 8 normalised floats. The agent sees everything except "
                "the draw pile *order*.\n\n"
                "This is a conscious design choice:\n"
                "- **Upside**: eliminates the need for belief tracking or memory (no LSTM/Transformer required), "
                "letting a simple feedforward MLP focus purely on strategic planning\n"
                "- **Downside**: the agent learns a different (easier) game — it can see opponents' hands, "
                "which real players cannot\n"
                "- **Justification**: experienced players can infer much of this — hands change visibly as "
                "cards are played and picked up, and card counting reveals the draw pile composition. "
                "The perfect-information version is a reasonable upper bound on what a strong player could deduce."
            ),

            wr.H2("Observation Space (156 floats for 3 players)"),
            M(
                "Two **discrete tokens** (embedded via learned 32-dim embeddings):\n"
                "- **phase** (0=play, 1=discard): tells the network which of the two completely different "
                "action subsets is currently active (100 play actions vs 8 discard actions)\n"
                "- **current_player**: always 0 under relative seat encoding\n\n"
                "**Continuous fields** (all normalised to [0, 1]):\n\n"
                "| Field | Dim | Purpose |\n"
                "|-------|-----|---------|\n"
                "| **hands** | 24 | All players' card holdings — for predicting opponents' plays |\n"
                "| **recycling_side** | 8 | Cards on factory recycling side — determines if threshold (value ≥ 4) to claim a factory card can be reached |\n"
                "| **waste_side** | 32 | Cards on each factory's waste side �� the player *must* pick these up when playing that color (the \"cost\" of each play) |\n"
                "| **factory_stacks** | 32 | Remaining factory card values per color — the reward landscape (what's the next claimable card worth?) |\n"
                "| **collected** | 24 | Per-player collected cards with scoring flag — a player needs *more than one* card of a color for it to score (1 card = 0 points) |\n"
                "| **penalty_pile** | 24 | Per-player penalties (-1 each; zero penalties earns a bonus equal to number of opponents with penalties) |\n"
                "| **scores** | 3 | Current normalised scores for relative standing |\n"
                "| **draw_pile_size** | 1 | Remaining deck size — proximity to reshuffling/game end |\n"
                "| **draw_pile_comp** | 8 | Remaining cards by (color, type) — for reasoning about future draw probabilities |"
            ),

            wr.H2("Relative Seat Encoding"),
            M(
                "All player-indexed observations are **rotated** so the agent always sees itself at index 0. "
                "\"My hand\" is always the first slot, \"my score\" is always `scores[0]`. This eliminates the need "
                "for the network to learn permutation invariance across seats, and makes the agent seat-agnostic "
                "by construction — important for self-play where the agent is randomly assigned to any seat each episode.\n\n"
                "Both absolute and relative encoding are valid approaches; we use relative encoding throughout as a standard default."
            ),

            wr.H2("Benchmarking Against a Random Agent"),
            M(
                "During training, we periodically evaluate against a **random opponent** (uniform over legal actions). "
                "While weak, this is a reliable proxy for monitoring progress:\n\n"
                "- **Monotonic signal**: unlike self-play metrics (which oscillate due to co-adaptation), win rate "
                "against a fixed random opponent increases monotonically — an unambiguous measure of improvement\n"
                "- **Cheap to compute**: 200 games with no gradients, negligible overhead\n"
                "- **Sensitive in early-to-mid training**: a random 3-player agent wins ~33% by chance. "
                "Basic strategy reaches 70-80%; the gap from 80% to 99% reflects increasingly sophisticated play "
                "(hand management, factory timing, opponent awareness)\n"
                "- **Mean score as secondary metric**: when win rate saturates near 99%, mean score still "
                "differentiates quality — a stronger agent wins by larger margins\n\n"
                "In cross-play between trained checkpoints, win rates hover near 33% (similar-strength agents in a "
                "symmetric game), making tournament results uninformative for tracking improvement. "
                "The random baseline avoids this by providing a fixed reference point."
            ),

            # ══════════════════════════════════════════════════════════════
            # ABLATION STUDY
            # ══════════════════════════════════════════════════════════════
            wr.H1("1. Ablation Study (10M Samples)"),
            M(
                "We isolate the effect of two techniques through four controlled experiments, each trained "
                "for **10M samples** with identical hyperparameters.\n\n"
                "**A note on explained variance:** This is the key diagnostic metric in this study. "
                "It measures how well the critic predicts actual returns: "
                "`1 - Var(returns - predictions) / Var(returns)`. Because we use **no reward shaping** "
                "(terminal +1/-1 only), the critic's value estimate *is* the learning signal — "
                "advantage estimates depend entirely on its accuracy. Low explained variance means "
                "noisy advantages and stalled training. In games with dense rewards, a poor critic is "
                "partially compensated by the immediate reward; here there is no such safety net.\n\n"
                "| Condition | Entropy Coef | GAE λ | Win Rate vs Random | Expl. Variance | Value Loss |\n"
                "|-----------|-------------|-------|-------------------|----------------|------------|\n"
                "| Baseline | 0 (none) | 1.0 | ~89% | ~0.16 | ~0.29 |\n"
                "| Entropy only | 0.1 → 0.01 | 1.0 | ~93% | ~0.20 | ~0.26 |\n"
                "| GAE only | 0 (none) | 0.85 | ~93% | ~0.39 | ~0.07 |\n"
                "| **Both** | **0.1 → 0.01** | **0.85** | **~99%** | **~0.45** | **~0.05** |"
            ),

            wr.H2("Win Rate and Score vs Random"),
            M(
                "The combined configuration dramatically outperforms all single-technique ablations. "
                "The baseline plateaus around 89% win rate; individual techniques each add ~4%; but their combination "
                "jumps to ~99% — the effect is **synergistic, not additive**.\n\n"
                "The **mean score** panel provides a complementary view: while win rate can saturate near 100%, "
                "mean score continues to differentiate agent quality — a better agent doesn't just win more often, "
                "it wins by larger margins.\n\n"
                "- **Entropy without GAE**: the agent explores diverse strategies, but the high-variance critic "
                "can't tell which are better. Exploration is partially wasted.\n"
                "- **GAE without entropy**: the critic learns accurate values, but the policy collapses before "
                "the critic converges. It ends up evaluating a narrow, suboptimal strategy.\n"
                "- **Both**: entropy keeps exploring until the GAE-improved critic converges; the accurate critic "
                "then guides exploration toward genuinely better strategies."
            ),
            wr.PanelGrid(
                runsets=[wr.Runset(entity=ENTITY, project=PROJECT, filters=FILTER_10M)],
                panels=[
                    make_panel("benchmark/vs_random/win_rate", "Win Rate vs Random Opponent"),
                    make_panel("benchmark/vs_random/mean_score", "Mean Score vs Random Opponent"),
                ],
            ),

            wr.H2("Value Learning Quality"),
            M(
                "**GAE λ=0.85 is the key driver of value learning.** With λ=1.0 and γ=1.0, "
                "GAE reduces to Monte Carlo returns — unbiased but high variance, because a single +1/-1 "
                "outcome must be attributed across ~20 actions per episode. λ=0.85 trades slight bias for "
                "dramatically lower variance.\n\n"
                "Result: **~0.39 explained variance** vs ~0.16 (baseline), and value loss drops 4×. "
                "This creates a positive feedback loop: better value estimates → lower-variance advantages → "
                "more stable policy updates → the critic can learn from more consistent behaviour."
            ),
            wr.PanelGrid(
                runsets=[wr.Runset(entity=ENTITY, project=PROJECT, filters=FILTER_10M)],
                panels=[
                    make_panel("losses/explained_variance", "Explained Variance"),
                    make_panel("losses/value_loss", "Value Loss"),
                ],
            ),

            wr.H2("Entropy and Exploration"),
            M(
                "Without entropy regularisation, the policy collapses early (entropy drops to 0.06-0.09 — "
                "nearly deterministic). **Entropy annealing** (0.1 → 0.01 over 40k steps) prevents this: "
                "the agent maintains meaningful stochasticity (0.10-0.32), adapting its play to different "
                "board states rather than following a single fixed strategy.\n\n"
                "The combined run maintains the highest entropy (~0.32), indicating a richer, more robust policy "
                "that hasn't collapsed to a single deterministic strategy."
            ),
            wr.PanelGrid(
                runsets=[wr.Runset(entity=ENTITY, project=PROJECT, filters=FILTER_10M)],
                panels=[make_panel("losses/entropy", "Policy Entropy")],
            ),

            # ══════════════════════════════════════════════════════════════
            # SCALE COMPARISON
            # ══════════════════════════════════════════════════════════════
            wr.H1("2. Training Scale: 10M vs 50M"),
            M(
                "Using the best configuration (entropy annealing + GAE=0.85), we compare **10M vs 50M training "
                "samples**. Does the agent plateau early, or does it keep improving with more compute?"
            ),

            wr.H2("Win Rate and Score Progression"),
            M(
                "Both achieve ~99% win rate vs random, but the **training curves** tell a richer story: "
                "the win rate **climbs steadily through the first 20M samples** before plateauing. "
                "The agent is learning meaningful strategy improvements well past 10M.\n\n"
                "The **mean score** curve is more informative at this saturation level — even when win rate "
                "is near-perfect, the agent continues to improve its score margin over time, indicating "
                "deeper strategic refinement.\n\n"
                "*Note: win rate vs random saturates at this level — mean score and stronger benchmarks "
                "are needed to differentiate agent quality further.*"
            ),
            wr.PanelGrid(
                runsets=[wr.Runset(entity=ENTITY, project=PROJECT, filters=FILTER_SCALE)],
                panels=[
                    make_panel("benchmark/vs_random/win_rate", "Win Rate vs Random: 10M vs 50M"),
                    make_panel("benchmark/vs_random/mean_score", "Mean Score vs Random: 10M vs 50M"),
                ],
            ),

            wr.H2("Continued Refinement"),
            M(
                "Entropy decreases from ~0.32 to ~0.20 over the 50M run, indicating the policy becomes more "
                "**decisive** — committing more strongly to preferred actions as confidence grows. "
                "Value metrics stabilise, suggesting the critic has converged.\n\n"
                "**Diminishing returns after ~30M samples**: all curves flatten. The agent has extracted most of "
                "the learnable signal from self-play at this difficulty level."
            ),
            wr.PanelGrid(
                runsets=[wr.Runset(entity=ENTITY, project=PROJECT, filters=FILTER_SCALE)],
                panels=[
                    make_panel("losses/explained_variance", "Explained Variance"),
                    make_panel("losses/value_loss", "Value Loss"),
                    make_panel("losses/entropy", "Policy Entropy"),
                    make_panel("losses/policy_loss", "Policy Loss"),
                ],
            ),

            # ══════════════════════════════════════════════════════════════
            # SUMMARY
            # ══════════════════════════════════════════════════════════════
            wr.H1("Summary"),
            M(
                "Two techniques transform a stagnating baseline (~89% win rate, ~0.16 explained variance) "
                "into a near-perfect agent (~99%, ~0.45 explained variance):\n\n"
                "1. **Entropy annealing** (0.1 → 0.01) prevents premature policy collapse\n"
                "2. **GAE λ = 0.85** enables accurate value learning despite terminal-only rewards\n\n"
                "Neither alone is sufficient — their **combination is critical** and synergistic. "
                "Combined with **adaptive LR via target KL** (Rudin et al., 2022), the agent reaches "
                "competence within 10M samples and saturates by ~30M."
            ),
        ],
    )

    report.save()
    print(f"Report URL: {report.url}")


if __name__ == "__main__":
    create_report()
