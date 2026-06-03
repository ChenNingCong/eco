# `abstract/` — Framework Architecture & Call Order

How the game-agnostic RL framework fits together, and the exact order methods
fire during construction, reset, stepping, and PPO training.

## Layer stack (bottom → top)

```
BaseGameEngine        game logic: _reset / step / legal_actions / encode / current_player / done   (game-specific)
  ↑ wrapped by
SinglePlayerEnv       multi-player engine → single-agent view; drives opponents between agent turns
  ↑ vectorized by
VecSinglePlayerEnv    N envs in lockstep; two interchangeable step paths (generic | BatchStepper)
  ↑ driven by
PPOLSTMTrainer        rollout → GAE → PPO update loop
```

Players are orthogonal to the stack:
- `BasePlayer.batch_action(obs, mask, idxs)` — batched opponent inference
- `BasePlayer.slice(i)` — per-env proxy (`SlicedPlayer`)
- `BasePlayer.reset(idxs)` — reset opponent-internal state (e.g. LSTM hidden) for given envs

In self-play the opponent is an `LSTMBatchedPlayer` wrapping the **same** network,
holding its own per-env LSTM state.

## RNG / construction order (`VecSinglePlayerEnv.__init__`)

1. `key.spawn(num_envs)` → one child `np.random.Generator` per env.
2. Each child `.spawn(2)` → `(engine_rng, env_rng)`.
3. `engine = factory.create(engine_rng)`; `SinglePlayerEnv(engine, opponent.slice(i), rng=env_rng)`.

Each env gets an independent stream.

> **Caveat:** the LC **jitclass** engine shuffles its deck via the *global* Numba RNG
> (`np.random.shuffle`/`np.random.randint`), **not** `engine_rng`. The post-reset seat
> is also drawn from the global RNG in the BatchStepper path (`batch_stepper.py:180`) but
> from the per-env `env_rng` in the generic path (`single_player_env.py:49`). The two step
> paths are therefore bit-identical *within* an episode but legitimately diverge *after* a
> reset (different but equally-valid random games) — this is benign, not a logic bug.

## `reset()`

### Generic path (post-fix order)
1. `env.reset()` for **every** env first. Each `SinglePlayerEnv.reset()`:
   `engine.reset()` → pick random seat from `env_rng` → advance opponents
   (`opponent.action`) until it is the agent's turn.
2. `_init_buffers()` — samples `engine.encode(0)` to size preallocated buffers.
   *(Envs must be reset before this — engines whose state is `None` pre-reset,
   e.g. r_eco / ttr, otherwise raise in `encode()`.)*
3. `encode_into` + `legal_actions` per env → return `(obs_buf, mask_buf)`.

### BatchStepper path
1. `env.reset()` per env (same `SinglePlayerEnv.reset()` as above), collecting `seats[]`.
2. `stepper.init(envs, seats)` — builds a `numba.typed.List` of jitclass states + preallocated buffers.
3. `stepper.encode_initial()` — JIT-encodes all agent obs/masks → return `stepper.obs, stepper.masks`.

## `step()` — the two paths

Both paths implement the same contract; Python only leaves JIT/loops for the
NN inference of opponent actions. Results are bit-identical within an episode.

### Generic (`_step_generic`) — pure-Python loop over `env.engine`
1. Validate each action against `legal_actions()`; illegal → first legal.
2. `engine.step(agent_action)` for all envs; accumulate reward at agent's seat.
3. **Opponent loop:** while any env has `current_player != seat` and not done →
   gather pending envs, `encode_into` / `legal_actions` for them, one
   `opponent.batch_action(...)`, then `engine.step(opp_action)` each. Repeat
   until all envs are back on the agent's turn.
4. **Collect:** per env record reward; if `done` → build `info`
   (`compute_scores`, `game_metrics`, terminal `encode`) then `env.reset()`;
   then `encode_into` + `legal_actions`.
5. Return `(obs, masks, rewards, terminated, truncated, infos)` — `infos` is a list of dicts.

### BatchStepper (`_step_stepper`) — same logic fused into `@njit`
1. `n_p = stepper.step_and_find_opponents(actions)` — JIT: validate + step agents
   + find pending opponents; writes opp obs/masks to internal buffers.
2. **Opponent loop:** while `n_p > 0`: `get_opponent_slice(n_p)` (views, no copy)
   → `opponent.batch_action(...)` → `n_p = apply_and_find_more(n_p, opp_actions)`.
3. `rewards, done_f32, done_indices, n_done = stepper.collect_and_encode()` — JIT:
   rewards + terminated + **reset done envs** (new seat via global RNG) + encode agent obs/masks.
4. **Post-reset opponent handling** (the structural difference vs generic): if
   `n_done > 0` → `handle_post_reset_opponents()` runs the opponent loop again for
   freshly-reset envs where the opponent moves first, then `re_encode_done(n_done)`
   re-encodes those agent obs.
5. Return `(stepper.obs, stepper.masks, rewards, terminated, truncated, stepper.info)`
   — `info` is a `StepInfo` of preallocated arrays, not dicts.

## PPO loop (`PPOLSTMTrainer`, per iteration)

1. Clone `initial_lstm_state`; anneal LR / entropy coefficient.
2. **Rollout**, for `step in range(num_steps)`:
   1. Store `next_obs` / `dones` / `masks` into rollout buffers.
   2. `agent.get_action_and_value(next_obs, masks, lstm_state, next_done)` →
      `action, logprob, value, next_lstm_state`.
   3. `envs.step(action.cpu().numpy())` → next obs/masks/reward/term/trunc/infos.
   4. `done_indices = where(term | trunc)`; if any → `opponent.reset(done_indices)`
      (zeroes the opponent's LSTM state for those envs).
   5. Refresh GPU `next_obs` / `next_masks` / `reward` / `next_done` — for the
      stepper path this is a **zero-copy** `copy_` from CPU tensors that alias the
      stepper's numpy buffers; for generic it rebuilds tensors from returned numpy.
   6. wandb-log episodic returns for `done_indices`.
3. **GAE** — `get_value` bootstrap, reversed advantage scan.
4. Flatten + minibatch PPO epochs (clip, value loss, entropy); optional adaptive
   entropy / target-KL LR control.
5. Periodic benchmark / checkpoint save.

## Notes & gotchas

- `SinglePlayerEnv.step` / `step_gen` exist but the vectorized **generic path does
  not use them** — it re-implements stepping directly over `env.engine` for batched
  opponent inference. Those methods are for non-vectorized / generator-based
  consumers (e.g. DQN, `mp_vec_env`).
- In self-play, `get_action_and_value` runs effectively **twice per env per step**:
  once for the agent (PPO loop) and once inside `batch_action` for the opponent —
  with separate LSTM state tensors maintained by `LSTMBatchedPlayer`.
- `encode()` should rotate player-indexed data so `player_id` sits at index 0, so the
  network never has to learn seat-awareness.
- `encode_into(player_id, out, idx)` has a default (encode + copy); override it for
  zero-alloc batched writes (LC's jitclass does).
