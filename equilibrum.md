# Technical Strategy 1: Linear Penalty Annealing (Curriculum Learning)

## Problem Addressed
**The "Pessimism Collapse."** In Lost Cities, the initial $-20$ cost is a catastrophic barrier. Standard RL agents quickly learn that "doing nothing" (0 points) is superior to "failing an expedition" (negative points), causing the policy to collapse into a permanent discard loop.

## Implementation Logic

The environment is treated as a Non-Stationary MDP where the reward function is a function of the training step $t$.

**The Linear Schedule:** Define the entry penalty $P(t)$ to scale from $0$ to $-20$ over a fixed horizon $T_{anneal}$ (e.g., $10^6$ steps).

$$P(t) = \text{clip}\left(P_{start} + \frac{t}{T_{anneal}} \times (P_{target} - P_{start}), \text{min}=P_{target}, \text{max}=P_{start}\right)$$

**Hierarchical Scaling:** Start investment cards as flat bonuses (e.g., $+5$). Gradually transition them into $2\times / 3\times$ multipliers only after the penalty $P(t)$ crosses a threshold (e.g., $-15$).

**PopArt / Reward Normalization:** As the global reward magnitude shifts from positive to deeply negative, use adaptive reward normalization to prevent the Critic's loss from exploding.

**Action Masking:** Enforce game legality (ascending card order) via hard-coding an action mask to prevent the agent from wasting exploration steps on invalid moves.

## Outcome

The agent first masters the mechanics of card sequences in a "forgiving" environment, then gradually learns the economic threshold required to justify an expedition as the penalty increases.

---

# Technical Strategy 2: Neural Fictitious Self-Play (NFSP)

## Problem Addressed
**"Strategy Cycling."** In competitive self-play, agents often "chase" each other's latest iterations (Aggressive $\to$ Defensive $\to$ Passive), failing to converge to a stable Nash Equilibrium (NE).

## Implementation Logic

NFSP approximates the game-theoretic concept of Fictitious Play by training the agent to play against the *average strategy* of the opponent.

**The Best Response (RL Head):** A DQN (or off-policy agent) that learns to maximize reward against an opponent who is acting according to their Average Policy.
- *Mechanism:* The RL head is trained on transitions where the opponent's actions were sampled from their SL Head.

**The Average Policy (SL Head):** A Supervised Learning network that learns to mimic the historical actions of the agent's own RL Head.
- *Data Source:* A Reservoir Sampling Buffer that stores $(state, action)$ pairs from the agent's own Best Response history. This ensures the SL head represents a uniform distribution of all strategies used across the training run.

**$\eta$-Greedy Action Selection:** During data generation, the agent chooses an action from its RL Head with probability $\eta$ (e.g., $0.1$) and its SL Head with probability $1-\eta$ (e.g., $0.9$).

## Algorithm 1: Neural Fictitious Self-Play (NFSP) with Fitted Q-Learning

Initialize game Γ and execute an agent via `RunAgent` for each player in the game.

---

### Function `RunAgent(Γ)`

**Initialize:**
- Replay memories $M_{RL}$ (circular buffer) and $M_{SL}$ (reservoir)
- Average-policy network $\Pi(s, a \mid \theta^\Pi)$ with random parameters $\theta^\Pi$
- Action-value network $Q(s, a \mid \theta^Q)$ with random parameters $\theta^Q$
- Target network parameters $\theta^{Q'} \leftarrow \theta^Q$
- Anticipatory parameter $\eta$

---

**For each episode do:**

1. Set policy:
$$\sigma \leftarrow \begin{cases} \varepsilon\text{-greedy}(Q) & \text{with probability } \eta \\ \Pi & \text{with probability } 1 - \eta \end{cases}$$

2. Observe initial information state $s_1$ and reward $r_1$

3. **For** $t = 1, \ldots, T$ **do:**

   - Sample action $a_t$ from policy $\sigma$
   - Execute action $a_t$ in game; observe reward $r_{t+1}$ and next state $s_{t+1}$
   - Store transition $(s_t, a_t, r_{t+1}, s_{t+1})$ in $M_{RL}$
   - **If** agent follows best response policy $\sigma = \varepsilon\text{-greedy}(Q)$:
     - Store behaviour tuple $(s_t, a_t)$ in $M_{SL}$
   - **End if**
   - Update $\theta^\Pi$ via SGD on loss:
$$\mathcal{L}(\theta^\Pi) = \mathbb{E}_{(s,a) \sim M_{SL}} \left[ -\log \Pi(s, a \mid \theta^\Pi) \right]$$
   - Update $\theta^Q$ via SGD on loss:
$$\mathcal{L}(\theta^Q) = \mathbb{E}_{(s,a,r,s') \sim M_{RL}} \left[ \left( r + \max_{a'} Q(s', a' \mid \theta^{Q'}) - Q(s, a \mid \theta^Q) \right)^2 \right]$$
   - Periodically update target network: $\theta^{Q'} \leftarrow \theta^Q$

   **End for**

**End for**

**End function**