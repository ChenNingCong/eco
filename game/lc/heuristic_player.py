"""
Heuristic ("mechanic") Lost Cities player — a rule-based *selective* strategy.

Plugs in as a BasePlayer/SlicedPlayer (same interface as RandomPlayer), so it can be
used as a PPO self-play opponent, a benchmark baseline, or a behavior-cloning teacher
to bootstrap RL past the investment "reward cliff".

The whole point is to play the way the RL agents fail to: COMMIT TO FEW COLORS and
build them deep, rather than opening all 5 lanes shallow.

Policy (greedy, per turn; one flat action = card + expedition/discard + draw source):
  - Open a new expedition only if we hold enough points in that color to clear the
    -20 penalty (hold_pts >= OPEN_THRESH); otherwise discard instead of over-opening.
  - Play investment (value-0) cards only at the very start of a STRONGLY committed
    color (hold_pts >= INVEST_THRESH and no number card placed yet).
  - Extend committed expeditions low-card-first (preserves the ascending sequence).
  - Discard the least useful card (low value / off-color / non-committed).
  - Draw from a discard pile only when its top extends a committed expedition, else deck.

Reads the batched LCObs the env feeds opponents (hand / own_expeditions / discard_top1).
"""
import numpy as np

from abstract.player import BasePlayer, SlicedPlayer

NUM_ACTIONS = 600

# Fixed action-index decode tables: action = card_id*12 + action_type*6 + draw_source
_A = np.arange(NUM_ACTIONS)
_ATYPE = (_A // 6) % 2          # 0 = play to expedition, 1 = discard
_DRAW = _A % 6                  # 0 = deck, 1..5 = draw from discard pile of color 0..4
_CC = (_A // 12) // 10          # color of the card
_VV = (_A // 12) % 10           # value_idx: 0 = investment, 1..9 = cards 2..10
_PTS = np.where(_VV == 0, 0, _VV + 1).astype(np.float32)  # point value of the card

# Card points by value_idx: idx 0 = investment (0 pts), idx 1..9 = cards 2..10.
_PTS_BY_V = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

# Tunables
OPEN_THRESH = 18.0    # min held points in a color before opening its expedition
INVEST_THRESH = 14.0  # min held points before risking investment multipliers
INVEST_MIN_CARDS = 3  # need enough number cards of the color to build it deep


def choose_action(hand, own_exp, dt1, mask) -> int:
    """Best legal action for one env. hand/own_exp/dt1 are (50,) LCObs fields
    (hand, own_exp are /3 counts; dt1 is one-hot top-of-pile); mask is (600,) bool."""
    hand = np.asarray(hand); own_exp = np.asarray(own_exp); dt1 = np.asarray(dt1)
    mask = np.asarray(mask, dtype=bool)

    hc = np.rint(hand * 3.0).astype(np.int32).reshape(5, 10)      # held card counts
    oc = np.rint(own_exp * 3.0).astype(np.int32).reshape(5, 10)   # own expedition cards
    d1 = dt1.reshape(5, 10)

    own_open = oc.sum(axis=1) > 0                                  # (5,) lane started?
    has_numbers = oc[:, 1:].sum(axis=1) > 0                        # (5,) any number placed?
    own_top = np.where(oc[:, 1:].any(axis=1),
                       9 - np.argmax(oc[:, 1:][:, ::-1] > 0, axis=1), 0)  # highest value_idx
    hold_pts = (hc[:, 1:] > 0) @ _PTS_BY_V[1:]                     # (5,) held points/color
    hold_cnt = (hc[:, 1:] > 0).sum(axis=1)                         # (5,) number cards held
    committed = own_open | (hold_pts >= OPEN_THRESH)              # open a lane only with material
    disc_top = np.where(d1.any(axis=1), np.argmax(d1 > 0, axis=1), 0)  # (5,) top value_idx

    cc, vv = _CC, _VV
    is_exp = _ATYPE == 0
    is_inv = vv == 0

    # PLAY score: extend open lanes; open a new lane only if it's profitable (hold_pts >=
    # OPEN_THRESH) — refusing to open shallow money-losing lanes is the whole point.
    base = np.where(own_open[cc], 30.0,
                    np.where(hold_pts[cc] >= OPEN_THRESH, 10.0, -50.0))
    play_num = base + (9 - vv)                                    # extend/open, low-first
    invest_ok = (~has_numbers[cc]) & (hold_pts[cc] >= INVEST_THRESH) & (hold_cnt[cc] >= INVEST_MIN_CARDS)
    play_inv = np.where(invest_ok, 60.0, -100.0)
    play_dis = -5.0 + np.where(committed[cc], -0.5 * _PTS, 0.2 * (10 - _PTS))
    play = np.where(is_exp & ~is_inv, play_num,
                    np.where(is_exp & is_inv, play_inv, play_dis))

    # DRAW score
    d = np.clip(_DRAW - 1, 0, 4)
    useful = (disc_top[d] >= 1) & committed[d] & (disc_top[d] > own_top[d])
    draw_sc = np.where(_DRAW == 0, 0.5, np.where(useful, 2.0, -1.0))

    score = np.where(mask, play + draw_sc, -1e9)
    return int(np.argmax(score))


class HeuristicPlayer(BasePlayer, SlicedPlayer):
    """Rule-based selective Lost Cities player. Stateless."""

    def batch_action(self, obs_batch, mask_batch, idxs=None) -> np.ndarray:
        mask_batch = np.asarray(mask_batch)
        n = mask_batch.shape[0]
        hand, own, dt1 = obs_batch.hand, obs_batch.own_expeditions, obs_batch.discard_top1
        out = np.empty(n, dtype=np.int32)
        for i in range(n):
            out[i] = choose_action(hand[i], own[i], dt1[i], mask_batch[i])
        return out

    def reset(self, env_indices=None) -> None:
        pass

    def slice(self, env_idx: int) -> "HeuristicPlayer":
        return self

    def action(self, obs, mask) -> int:
        return choose_action(obs.hand, obs.own_expeditions, obs.discard_top1, mask)
