"""
Vectorized R-öko environment — single-process, generator-based batch stepping.

Key idea (from async.md):
  Instead of evaluating the opponent NN once per game per opponent turn (N×1
  forward passes), all N games yield their pending opponent observations, the
  caller evaluates the NN in a single batched call, and sends actions back.
  This turns O(N) NN calls into O(rounds) NN calls each with batch size ≈ N.

VecSinglePlayerEcoEnv runs N independent SinglePlayerEcoEnv instances.
Each env's step() is a generator (step_gen); the vec env drives all generators
in lockstep, collecting opponent obs across games before each batched NN call.

Interface:
  obs, masks = vec.reset()
  obs, masks, rewards, terminated, truncated, infos = vec.step(actions, batch_opponent_fn)

  batch_opponent_fn : callable(EcoPyTreeObs, (N,108) ndarray) -> (N,) int ndarray
      If None, falls back to each env's per-game opponent_fn (backward compat).
"""

import copy
import pickle
import traceback

import numpy as np
from typing import List, Optional, Tuple, Callable

from eco_obs_encoder import SinglePlayerEcoEnv, EcoPyTreeObs
from eco_env import NUM_ACTIONS


def _stack_obs(obs_list: List[EcoPyTreeObs]) -> EcoPyTreeObs:
    return EcoPyTreeObs(**{
        field: np.stack([getattr(o, field) for o in obs_list], axis=0)
        for field in EcoPyTreeObs._fields
    })


class VecSinglePlayerEcoEnv:
    """
    Synchronous vectorised single-player R-öko environment.

    Parameters
    ----------
    num_envs             : int
    num_players          : int              (default 2)
    opponent             : BatchedPlayer, callable, or None
        - BatchedPlayer (or compatible with batch_action/slice): each env gets
          opponent.slice(i).action for reset(), step() uses batch_action().
        - callable(obs, mask) -> int: used as per-env opponent_fn directly.
        - None: random opponent (default from SinglePlayerEcoEnv).
    seeds                : list of int, optional
    reward_shaping_scale : float            (default 1.0)
    """

    def __init__(
        self,
        num_envs: int,
        num_players: int = 2,
        opponent = None,
        seeds: Optional[List[int]] = None,
        reward_shaping_scale: float = 1.0,
        opponent_penalty: float = 0.5,
        relative_seat: bool = True,
    ):
        self.num_envs = num_envs
        self._step_count = 0
        assert opponent is not None and hasattr(opponent, 'batch_action'), \
            "opponent must be a BasePlayer with batch_action/slice interface"
        self._batched_opponent = opponent
        _get_fn = lambda i: opponent.slice(i).action
        seeds = seeds or [None] * num_envs
        self._init_seeds = list(seeds)
        self.envs: List[SinglePlayerEcoEnv] = [
            SinglePlayerEcoEnv(
                num_players=num_players,
                opponent_fn=_get_fn(i),
                seed=s,
                reward_shaping_scale=reward_shaping_scale,
                opponent_penalty=opponent_penalty,
                relative_seat=relative_seat,
            )
            for i, s in enumerate(seeds)
        ]

    def reset(self, seed: Optional[int] = None) -> Tuple[EcoPyTreeObs, np.ndarray]:
        if seed is not None:
            for i, env in enumerate(self.envs):
                env.env.rng = np.random.default_rng(seed + i)

        obs_list, masks = [], []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
            masks.append(env.legal_actions())
        return _stack_obs(obs_list), np.stack(masks, axis=0)

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[EcoPyTreeObs, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Vectorised step using generator-based batched opponent evaluation.

        If self._batched_opponent is set, pending opponent observations are collected
        and evaluated in a single batch_action() call per round.
        Otherwise, each env's per-game opponent_fn is used (one call per game).

        Parameters
        ----------
        actions : (N,) int  agent actions for each env

        Returns
        -------
        obs        : EcoPyTreeObs     batched next agent observation
        masks      : (N, NUM_ACTIONS) bool
        rewards    : (N,)             float32
        terminated : (N,)             bool
        truncated  : (N,)             bool  (always False)
        infos      : list of dicts
        """
        assert len(actions) == self.num_envs
        N = self.num_envs
        self._step_count += 1

        # ── 1. Start a generator for every env ───────────────────────────────
        # Snapshot state before stepping for post-mortem debugging
        _pre_states = [copy.deepcopy(env.env.state) for env in self.envs]
        _pre_seats  = [env._seat for env in self.envs]

        for i, (env, a) in enumerate(zip(self.envs, actions)):
            mask = env.legal_actions()
            if not mask[int(a)]:
                # Hard enforcement: pick first legal action instead of crashing
                legal = np.where(mask)[0]
                fallback = int(legal[0]) if len(legal) > 0 else 0
                print(f"[WARN] VecEnv hard mask enforcement: env {i}, "
                      f"action={int(a)}→{fallback}, phase={env.state.phase}, "
                      f"player={env.state.current_player}, seat={env._seat}")
                actions[i] = fallback
        gens    = [env.step_gen(int(a)) for env, a in zip(self.envs, actions)]
        results = [None] * N   # will hold (obs, rew, term, trunc, info) when done

        # ── 2. Prime all generators (advance to first yield or completion) ───
        pending: dict[int, tuple] = {}   # env_idx -> (opp_obs, opp_mask)
        for i in range(N):
            try:
                opp_obs, opp_mask = next(gens[i])
                pending[i] = (opp_obs, opp_mask)
            except StopIteration as e:
                results[i] = e.value

        # ── 3. Drive generators until all are exhausted ──────────────────────
        #   Each "round" collects pending obs from all still-active games,
        #   evaluates one batched NN call, and sends actions back.
        while pending:
            idxs      = list(pending.keys())
            obs_list  = [pending[i][0] for i in idxs]
            mask_list = [pending[i][1] for i in idxs]

            # Batched opponent evaluation
            obs_batch  = _stack_obs(obs_list)
            mask_batch = np.stack(mask_list, axis=0)
            opp_actions = self._batched_opponent.batch_action(obs_batch, mask_batch, idxs)

            # Send actions back; collect next yield or final result
            new_pending: dict[int, tuple] = {}
            for j, i in enumerate(idxs):
                try:
                    opp_obs, opp_mask = gens[i].send(int(opp_actions[j]))
                    new_pending[i] = (opp_obs, opp_mask)
                except StopIteration as e:
                    results[i] = e.value
            pending = new_pending

        # ── 4. Auto-reset terminated envs and build output ───────────────────
        obs_list_out = []
        masks_out    = []
        rewards      = np.zeros(N, dtype=np.float32)
        terminated   = np.zeros(N, dtype=bool)
        truncated    = np.zeros(N, dtype=bool)
        infos        = []

        for i in range(N):
            obs, rew, term, trunc, info = results[i]
            if term:
                info["terminal_obs"] = obs
                obs, _ = self.envs[i].reset()
            obs_list_out.append(obs)
            masks_out.append(self.envs[i].legal_actions())
            rewards[i]    = float(rew)
            terminated[i] = term
            infos.append(info)

        return (_stack_obs(obs_list_out), np.stack(masks_out, axis=0),
                rewards, terminated, truncated, infos)

    def get_action_masks(self) -> np.ndarray:
        return np.stack([e.legal_actions() for e in self.envs], axis=0)

    def close(self):
        pass
