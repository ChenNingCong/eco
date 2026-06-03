"""
Lost Cities BatchStepper — fused JIT functions for high-throughput env stepping.

ALL per-env logic runs in @njit:
  - Action validation
  - Agent stepping
  - Opponent finding + obs encoding
  - Opponent stepping
  - Result collection + game metrics + engine reset
  - Agent obs encoding

Only escapes to Python for neural network inference.
Per-env rbuf_all is thread-safe for future prange (when num_envs >> 10k).
"""

import numpy as np
import numba as nb
from numba import prange
from numba.typed import List as NbList

from abstract.batch_stepper import BatchStepper
from .engine import LCObs, NUM_ACTIONS, NUM_COLORS


# ── StepInfo: fixed-layout replacement for Python dict infos ───────────────

class StepInfo:
    """Pre-allocated arrays for step results. Replaces list-of-dicts."""

    __slots__ = ('agent_seat', 'final_scores', 'total_points',
                 'num_investment', 'num_played', 'open_lanes',
                 'cards_in_hand', 'discard_draws', 'is_done',
                 'terminal_obs')

    def __init__(self, N: int):
        self.agent_seat = np.zeros(N, dtype=np.int8)
        self.final_scores = np.zeros((N, 2), dtype=np.float32)
        self.is_done = np.zeros(N, dtype=np.bool_)
        self.total_points = np.zeros(N, dtype=np.float32)
        self.num_investment = np.zeros(N, dtype=np.float32)
        self.num_played = np.zeros(N, dtype=np.float32)
        self.open_lanes = np.zeros(N, dtype=np.float32)
        self.cards_in_hand = np.zeros(N, dtype=np.float32)
        self.discard_draws = np.zeros(N, dtype=np.float32)
        self.terminal_obs = _alloc_lc_obs(N)


# ── Fused @njit functions ──────────────────────────────────────────────────

@nb.njit(cache=True)
def _fused_step_and_find(states, seats, actions, acc_rewards, rbuf_all, masks,
                         opp_hand, opp_own_exp, opp_opp_exp,
                         opp_dt1, opp_dt2, opp_dt3,
                         opp_deck, opp_scores, opp_phase,
                         opp_pc, opp_pt,
                         opp_mask, pending):
    """Validate actions + step all agents + find/encode opponents. One JIT call."""
    N = len(states)
    for i in range(N):
        acc_rewards[i] = 0.0
        s = states[i]
        if s.is_done:
            continue
        a = nb.int32(actions[i])
        if not masks[i, a]:
            a = nb.int32(0)
            for j in range(masks.shape[1]):
                if masks[i, j]:
                    a = nb.int32(j)
                    break
            actions[i] = a
        s.step_flat(a, rbuf_all[i])
        acc_rewards[i] = rbuf_all[i, nb.int32(seats[i])]

    n_p = nb.int32(0)
    for i in range(N):
        s = states[i]
        if s.is_done or s.current_player == seats[i]:
            continue
        pending[n_p] = nb.int32(i)
        p = s.current_player
        s.encode_obs(p, opp_hand[n_p], opp_own_exp[n_p], opp_opp_exp[n_p],
                     opp_dt1[n_p], opp_dt2[n_p], opp_dt3[n_p],
                     opp_deck[n_p], opp_scores[n_p], opp_phase[n_p],
                     opp_pc[n_p], opp_pt[n_p])
        s.fill_legal_flat(opp_mask[n_p])
        n_p += 1
    return n_p


@nb.njit(cache=True)
def _fused_apply_and_find(states, seats, pending, n_pending, opp_actions,
                          acc_rewards, rbuf_all,
                          opp_hand, opp_own_exp, opp_opp_exp,
                          opp_dt1, opp_dt2, opp_dt3,
                          opp_deck, opp_scores, opp_phase,
                          opp_pc, opp_pt, opp_mask):
    """Apply opponent actions + find more pending opponents. One JIT call."""
    for j in range(n_pending):
        i = pending[j]
        s = states[i]
        s.step_flat(nb.int32(opp_actions[j]), rbuf_all[i])
        acc_rewards[i] += rbuf_all[i, nb.int32(seats[i])]

    n_p = nb.int32(0)
    N = len(states)
    for i in range(N):
        s = states[i]
        if s.is_done or s.current_player == seats[i]:
            continue
        pending[n_p] = nb.int32(i)
        p = s.current_player
        s.encode_obs(p, opp_hand[n_p], opp_own_exp[n_p], opp_opp_exp[n_p],
                     opp_dt1[n_p], opp_dt2[n_p], opp_dt3[n_p],
                     opp_deck[n_p], opp_scores[n_p], opp_phase[n_p],
                     opp_pc[n_p], opp_pt[n_p])
        s.fill_legal_flat(opp_mask[n_p])
        n_p += 1
    return n_p


@nb.njit(parallel=True, cache=True)
def _parallel_collect_reset_encode(states, seats, acc_rewards,
                                   hand, own_exp, opp_exp,
                                   dt1, dt2, dt3,
                                   deck, scores, phase,
                                   pc, pt, mask,
                                   rewards_out, done_f32,
                                   info_seat, info_scores,
                                   info_total_pts, info_inv, info_played,
                                   info_lanes, info_hand, info_dd,
                                   info_done,
                                   t_hand, t_own_exp, t_opp_exp,
                                   t_dt1, t_dt2, t_dt3,
                                   t_deck, t_scores, t_phase,
                                   t_pc, t_pt):
    """Collect + metrics + reset + encode — parallel per env.

    done_indices packing is done separately (sequential) since building a
    packed list requires serial coordination.
    """
    N = len(states)
    for i in prange(N):
        s = states[i]
        rewards_out[i] = nb.float32(acc_rewards[i])

        if s.is_done:
            done_f32[i] = nb.float32(1.0)
            info_done[i] = True
            info_seat[i] = seats[i]
            info_scores[i, 0] = s.scores[0]
            info_scores[i, 1] = s.scores[1]

            p = nb.int32(seats[i])
            info_total_pts[i] = s.scores[p]
            n_inv = nb.int32(0)
            n_played = nb.int32(0)
            n_lanes = nb.int32(0)
            for c in range(5):
                n = s.exp_len[p, c]
                if n > 0:
                    n_lanes += 1
                    n_played += n
                    for jj in range(n):
                        if s.exp_vals[p, c, jj] == 0:
                            n_inv += 1
            info_inv[i] = nb.float32(n_inv)
            info_played[i] = nb.float32(n_played)
            info_lanes[i] = nb.float32(n_lanes)
            info_hand[i] = nb.float32(s.hands[p].sum())
            info_dd[i] = nb.float32(s.discard_draws[p])

            s.encode_obs(p, t_hand[i], t_own_exp[i], t_opp_exp[i],
                         t_dt1[i], t_dt2[i], t_dt3[i],
                         t_deck[i], t_scores[i], t_phase[i],
                         t_pc[i], t_pt[i])

            # Reset engine in JIT (thread-local Numba PRNG)
            s.reset()
            seats[i] = nb.int8(np.random.randint(0, 2))
        else:
            done_f32[i] = nb.float32(0.0)
            info_done[i] = False

        # encode_obs + fill_legal_flat is the expensive part (~1μs/env)
        s.encode_obs(nb.int32(seats[i]), hand[i], own_exp[i], opp_exp[i],
                     dt1[i], dt2[i], dt3[i],
                     deck[i], scores[i], phase[i],
                     pc[i], pt[i])
        s.fill_legal_flat(mask[i])


@nb.njit(cache=True)
def _pack_done_indices(done_f32, done_indices):
    """Pack done env indices (sequential, fast scan)."""
    n_done = nb.int32(0)
    for i in range(len(done_f32)):
        if done_f32[i] > nb.float32(0.5):
            done_indices[n_done] = nb.int32(i)
            n_done += 1
    return n_done


@nb.njit(cache=True)
def _find_opponents(states, seats, pending,
                    opp_hand, opp_own_exp, opp_opp_exp,
                    opp_dt1, opp_dt2, opp_dt3,
                    opp_deck, opp_scores, opp_phase,
                    opp_pc, opp_pt, opp_mask):
    """Find envs needing opponent turn + encode their obs."""
    n_p = nb.int32(0)
    for i in range(len(states)):
        s = states[i]
        if s.is_done or s.current_player == seats[i]:
            continue
        pending[n_p] = nb.int32(i)
        p = s.current_player
        s.encode_obs(p, opp_hand[n_p], opp_own_exp[n_p], opp_opp_exp[n_p],
                     opp_dt1[n_p], opp_dt2[n_p], opp_dt3[n_p],
                     opp_deck[n_p], opp_scores[n_p], opp_phase[n_p],
                     opp_pc[n_p], opp_pt[n_p])
        s.fill_legal_flat(opp_mask[n_p])
        n_p += 1
    return n_p


@nb.njit(cache=True)
def _re_encode_done(states, seats, done_indices, n_done,
                    hand, own_exp, opp_exp,
                    dt1, dt2, dt3,
                    deck, scores, phase,
                    pc, pt, mask):
    """Re-encode agent obs for done envs after post-reset opponent turns."""
    for j in range(n_done):
        i = done_indices[j]
        s = states[i]
        p = nb.int32(seats[i])
        s.encode_obs(p, hand[i], own_exp[i], opp_exp[i],
                     dt1[i], dt2[i], dt3[i],
                     deck[i], scores[i], phase[i],
                     pc[i], pt[i])
        s.fill_legal_flat(mask[i])


@nb.njit(cache=True)
def _batch_encode_all(states, seats,
                      hand, own_exp, opp_exp,
                      dt1, dt2, dt3,
                      deck, scores, phase,
                      pc, pt, mask):
    """Encode agent obs + masks for all envs."""
    for i in range(len(states)):
        s = states[i]
        p = nb.int32(seats[i])
        s.encode_obs(p, hand[i], own_exp[i], opp_exp[i],
                     dt1[i], dt2[i], dt3[i],
                     deck[i], scores[i], phase[i],
                     pc[i], pt[i])
        s.fill_legal_flat(mask[i])


# ── Helpers ────────────────────────────────────────────────────────────────

def _alloc_lc_obs(N: int) -> LCObs:
    return LCObs(
        hand=np.zeros((N, 50), dtype=np.float32),
        own_expeditions=np.zeros((N, 50), dtype=np.float32),
        opp_expeditions=np.zeros((N, 50), dtype=np.float32),
        discard_top1=np.zeros((N, 50), dtype=np.float32),
        discard_top2=np.zeros((N, 50), dtype=np.float32),
        discard_top3=np.zeros((N, 50), dtype=np.float32),
        deck_size=np.zeros((N, 1), dtype=np.float32),
        scores=np.zeros((N, 2), dtype=np.float32),
        phase=np.zeros((N, 1), dtype=np.int32),
        played_card=np.zeros((N, 50), dtype=np.float32),
        played_type=np.zeros((N, 1), dtype=np.float32),
    )


def _obs_fields(obs: LCObs):
    return (obs.hand, obs.own_expeditions, obs.opp_expeditions,
            obs.discard_top1, obs.discard_top2, obs.discard_top3,
            obs.deck_size, obs.scores, obs.phase,
            obs.played_card, obs.played_type)


# ── LCBatchStepper ─────────────────────────────────────────────────────────

class LCBatchStepper(BatchStepper):
    """Lost Cities batch stepper with fused JIT functions."""

    def init(self, envs, seats):
        N = len(envs)
        self._N = N
        self._states = NbList()
        for env in envs:
            self._states.append(env.engine._state)
        self._seats = seats
        self._rbuf_all = np.zeros((N, 2), dtype=np.float64)
        self._acc_rewards = np.zeros(N, dtype=np.float64)
        self._obs_buf = _alloc_lc_obs(N)
        self._mask_buf = np.zeros((N, NUM_ACTIONS), dtype=np.bool_)
        self._opp_obs = _alloc_lc_obs(N)
        self._opp_mask = np.zeros((N, NUM_ACTIONS), dtype=np.bool_)
        self._pending = np.zeros(N, dtype=np.int32)
        self._rewards = np.zeros(N, dtype=np.float32)
        self._done_f32 = np.zeros(N, dtype=np.float32)
        self._done_indices = np.zeros(N, dtype=np.int32)
        self.info = StepInfo(N)

    def step_and_find_opponents(self, actions):
        return _fused_step_and_find(
            self._states, self._seats, actions,
            self._acc_rewards, self._rbuf_all, self._mask_buf,
            *_obs_fields(self._opp_obs),
            self._opp_mask, self._pending)

    def get_opponent_slice(self, n_pending):
        o = self._opp_obs
        obs_slice = LCObs(
            hand=o.hand[:n_pending],
            own_expeditions=o.own_expeditions[:n_pending],
            opp_expeditions=o.opp_expeditions[:n_pending],
            discard_top1=o.discard_top1[:n_pending],
            discard_top2=o.discard_top2[:n_pending],
            discard_top3=o.discard_top3[:n_pending],
            deck_size=o.deck_size[:n_pending],
            scores=o.scores[:n_pending],
            phase=o.phase[:n_pending],
            played_card=o.played_card[:n_pending],
            played_type=o.played_type[:n_pending],
        )
        indices = [int(self._pending[j]) for j in range(n_pending)]
        return obs_slice, self._opp_mask[:n_pending], indices

    def apply_and_find_more(self, n_pending, opp_actions):
        return _fused_apply_and_find(
            self._states, self._seats, self._pending, n_pending,
            opp_actions, self._acc_rewards, self._rbuf_all,
            *_obs_fields(self._opp_obs),
            self._opp_mask)

    def collect_and_encode(self):
        info = self.info
        # Phase 1: parallel collect/reset/encode (heavy: encode_obs for all envs)
        _parallel_collect_reset_encode(
            self._states, self._seats, self._acc_rewards,
            *_obs_fields(self._obs_buf), self._mask_buf,
            self._rewards, self._done_f32,
            info.agent_seat, info.final_scores,
            info.total_points, info.num_investment, info.num_played,
            info.open_lanes, info.cards_in_hand, info.discard_draws,
            info.is_done,
            *_obs_fields(info.terminal_obs))
        # Phase 2: sequential done_indices packing (fast scan)
        n_done = _pack_done_indices(self._done_f32, self._done_indices)
        return self._rewards, self._done_f32, self._done_indices, n_done

    def handle_post_reset_opponents(self):
        return _find_opponents(
            self._states, self._seats, self._pending,
            *_obs_fields(self._opp_obs),
            self._opp_mask)

    def re_encode_done(self, n_done):
        _re_encode_done(
            self._states, self._seats, self._done_indices, n_done,
            *_obs_fields(self._obs_buf), self._mask_buf)

    def encode_initial(self):
        _batch_encode_all(
            self._states, self._seats,
            *_obs_fields(self._obs_buf), self._mask_buf)

    @property
    def obs(self):
        return self._obs_buf

    @property
    def masks(self):
        return self._mask_buf
