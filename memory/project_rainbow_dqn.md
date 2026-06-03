---
name: Rainbow DQN for sparse rewards
description: Plan to implement Rainbow DQN to fix sparse reward credit assignment in LC DQN training
type: project
---

Rainbow DQN needed to make sparse/terminal-only rewards work for Lost Cities DQN.

**Why:** Terminal-only rewards (rawscore, zeroone, default) perform terribly with vanilla DQN — rawscore scores -54 vs random at 7M steps. The 40-step bootstrapping chain with gamma=1.0 can't propagate signal, and max overestimation compounds over the chain. Dense reward variants work well (score +22 vs random) because they provide per-step credit. PPO handles sparse rewards with low GAE lambda because V(s')-V(s) acts as learned dense shaping — DQN lacks this stability.

**How to apply:** Implement Rainbow DQN components, prioritized by impact:
1. **n-step returns** (most critical) — shortens the bootstrapping chain, propagates terminal reward to last n transitions directly
2. **Double DQN** — reduces max overestimation bias that compounds over 40 steps
3. **Prioritized experience replay** — samples terminal-adjacent transitions more, accelerating signal propagation
4. **Dueling architecture** — separates state value from action advantage, better for sparse rewards
5. **Distributional RL (C51)** — models return distribution, more stable with sparse signal
6. **Noisy nets** — replaces epsilon-greedy exploration

Start with n-step + Double DQN as minimum viable improvement. Full Rainbow if those show promise.

**Reference:** The existing DQN trainer is at `abstract/dqn.py` (DQNTrainer), LC-specific at `game/lc/train_dqn.py`.
