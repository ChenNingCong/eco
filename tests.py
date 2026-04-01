"""
Tests for the Hearts RL implementation.

Covers:
  hearts_env       : constants, reset, legal actions, game completion, rewards
  obs_encoder      : PyTreeObs structure, SinglePlayerEnv correctness
  vec_env          : VecSinglePlayerEnv shapes, auto-reset, reproducibility
  Agent            : forward shapes, masking, gradient flow   (requires torch)
  Integration      : pytree storage roundtrip                 (requires torch)

Run:  python -m pytest tests.py -v
  or: python tests.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

from hearts_env import (
    HeartsEnv, HeartsState, RoundRecord,
    NUM_CARDS, NUM_PLAYERS, NUM_ROUNDS, MAX_SCORE,
    card_suit, card_rank, card_points, make_card,
    TWO_OF_CLUBS, QUEEN_OF_SPADES,
    HEARTS, CLUBS, DIAMONDS, SPADES,
)
from obs_encoder import (
    SinglePlayerEnv, PyTreeObs,
    PAD_TOKEN, PLAYER_OFFSET, CARD_OFFSET,
    NUM_PLAYER_TOKENS, NUM_CARD_TOKENS,
    _encode_trick,
    HeartsEnvWrapper,   # alias
)
from vec_env import VecSinglePlayerEnv, VecHeartsEnv


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_game_raw(seed=0) -> HeartsState:
    env = HeartsEnv(seed=seed)
    rng = np.random.default_rng(seed)
    env.reset()
    while not env.state.done:
        env.step(rng.choice(np.where(env.legal_actions())[0]))
    return env.state


def _play_single_full(seed=0):
    """Play one full episode through SinglePlayerEnv with random moves everywhere."""
    env = SinglePlayerEnv(seed=seed)
    rng = np.random.default_rng(seed)
    env.reset()
    steps = []
    while True:
        mask   = env.legal_actions()
        action = rng.choice(np.where(mask)[0])
        result = env.step(action)
        steps.append(result)
        if result[2]:   # terminated
            break
    return steps


EXPECTED_FIELDS = {
    "history_leading"      : ((13,),       np.int32),
    "history_pairs"        : ((13, 4, 2),  np.int32),
    "current_trick_leading": ((1,),        np.int32),
    "current_trick_pairs"  : ((4, 2),      np.int32),
    "current_player"       : ((1,),        np.int32),
    "scores"               : ((4,),        np.float32),
    "round"                : ((1,),        np.float32),
    "hand"                 : ((NUM_CARDS,),np.float32),
    "played"               : ((NUM_CARDS,),np.float32),
    "leading_suit"         : ((4,),        np.float32),
    "hearts_broken"        : ((1,),        np.float32),
}


# ══════════════════════════════════════════════════════════════════════════════
# hearts_env: constants
# ══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_two_of_clubs(self):
        assert TWO_OF_CLUBS == 0
        assert card_suit(TWO_OF_CLUBS) == CLUBS
        assert card_rank(TWO_OF_CLUBS) == 0

    def test_queen_of_spades(self):
        assert card_suit(QUEEN_OF_SPADES) == SPADES
        assert card_rank(QUEEN_OF_SPADES) == 10
        assert card_points(QUEEN_OF_SPADES) == 13

    def test_all_hearts_score_1(self):
        for r in range(13):
            assert card_points(make_card(HEARTS, r)) == 1

    def test_non_scoring_cards(self):
        assert card_points(make_card(CLUBS,    5)) == 0
        assert card_points(make_card(DIAMONDS, 7)) == 0
        assert card_points(make_card(SPADES,   3)) == 0

    def test_total_points(self):
        assert sum(card_points(c) for c in range(NUM_CARDS)) == MAX_SCORE

    def test_action_space_is_52(self):
        """No NO_OP action — action space is exactly 52 cards."""
        assert NUM_CARDS == 52


# ══════════════════════════════════════════════════════════════════════════════
# hearts_env: game mechanics
# ══════════════════════════════════════════════════════════════════════════════

class TestHeartsEnvReset:
    def test_each_player_gets_13_cards(self):
        env = HeartsEnv(seed=0)
        s = env.reset()
        for p in range(NUM_PLAYERS):
            assert s.hands[p].sum() == NUM_ROUNDS

    def test_no_duplicate_cards(self):
        s = HeartsEnv(seed=0).reset()
        assert (s.hands.sum(axis=0) == 1).all()

    def test_starting_player_holds_two_of_clubs(self):
        for seed in range(10):
            s = HeartsEnv(seed=seed).reset()
            assert s.hands[s.current_player, TWO_OF_CLUBS]

    def test_initial_scores_zero(self):
        assert (HeartsEnv(seed=0).reset().scores == 0).all()

    def test_hearts_not_broken_at_start(self):
        assert not HeartsEnv(seed=0).reset().hearts_broken


class TestLegalActions:
    def test_first_must_be_two_of_clubs(self):
        env = HeartsEnv(seed=7)
        env.reset()
        mask = env.legal_actions()
        assert mask.sum() == 1 and mask[TWO_OF_CLUBS]

    def test_must_follow_suit(self):
        for seed in range(15):
            env = HeartsEnv(seed=seed)
            env.reset()
            rng = np.random.default_rng(seed)
            while not env.state.done:
                s    = env.state
                mask = env.legal_actions()
                if s.current_trick_count > 0:
                    lead = card_suit(s.current_trick_cards[0])
                    hand_suits = {card_suit(c) for c in range(NUM_CARDS)
                                  if s.hands[s.current_player, c]}
                    if lead in hand_suits:
                        for c in range(NUM_CARDS):
                            if mask[c]:
                                assert card_suit(c) == lead
                env.step(rng.choice(np.where(mask)[0]))

    def test_legal_subset_of_hand(self):
        env = HeartsEnv(seed=1)
        env.reset()
        rng = np.random.default_rng(1)
        while not env.state.done:
            s    = env.state
            mask = env.legal_actions()
            assert not (mask & ~s.hands[s.current_player]).any()
            env.step(rng.choice(np.where(mask)[0]))

    def test_hearts_not_led_until_broken(self):
        for seed in range(10):
            env = HeartsEnv(seed=seed)
            env.reset()
            rng = np.random.default_rng(seed)
            while not env.state.done and not env.state.hearts_broken:
                s = env.state
                if s.current_trick_count == 0:
                    mask     = env.legal_actions()
                    non_heart = s.hands[s.current_player].copy()
                    non_heart[HEARTS * 13:(HEARTS + 1) * 13] = False
                    if non_heart.any():
                        for c in range(NUM_CARDS):
                            if mask[c]:
                                assert card_suit(c) != HEARTS
                env.step(rng.choice(np.where(env.legal_actions())[0]))


class TestGameCompletion:
    def test_ends_after_13_tricks(self):
        s = _random_game_raw()
        assert s.done and s.round_num == NUM_ROUNDS

    def test_all_cards_played(self):
        s = _random_game_raw(seed=5)
        assert s.played_cards.all() and not s.hands.any()

    def test_scores_sum_to_max_score(self):
        for seed in range(20):
            assert _random_game_raw(seed=seed).scores.sum() == MAX_SCORE

    def test_history_length(self):
        assert len(_random_game_raw(seed=99).history) == NUM_ROUNDS

    def test_trick_winners_valid(self):
        for r in _random_game_raw(seed=11).history:
            assert 0 <= r.winner < NUM_PLAYERS


class TestTerminalRewards:
    def test_normal_game(self):
        env = HeartsEnv(seed=0)
        env.reset()
        env.state.scores = np.array([10, 6, 4, 6], dtype=np.int32)
        r = env._terminal_rewards(env.state)
        np.testing.assert_allclose(r, -(np.array([10, 6, 4, 6]) / MAX_SCORE))

    def test_shoot_the_moon(self):
        env = HeartsEnv(seed=0)
        env.reset()
        env.state.scores = np.array([26, 0, 0, 0], dtype=np.int32)
        r = env._terminal_rewards(env.state)
        assert r[0] == -1.0 and (r[1:] == 1.0).all()


# ══════════════════════════════════════════════════════════════════════════════
# obs_encoder: _encode_trick
# ══════════════════════════════════════════════════════════════════════════════

class TestEncodeTrick:
    def test_none_returns_pad(self):
        tok, pairs = _encode_trick(None)
        assert tok == PAD_TOKEN
        assert pairs.shape == (NUM_PLAYERS, 2) and (pairs == 0).all()

    def test_complete_trick(self):
        rec = RoundRecord(
            leading_player=2,
            cards=np.array([5, 10, 20, 49], dtype=np.int32),
            players=np.array([2, 3, 0, 1],  dtype=np.int32),
            winner=2,
        )
        tok, pairs = _encode_trick(rec)
        assert tok == 2 + PLAYER_OFFSET
        for i, (c, p) in enumerate(zip([5, 10, 20, 49], [2, 3, 0, 1])):
            assert pairs[i, 0] == c + CARD_OFFSET
            assert pairs[i, 1] == p + PLAYER_OFFSET

    def test_partial_trick_pads_missing(self):
        rec = RoundRecord(
            leading_player=1,
            cards=np.array([5, -1, -1, -1],   dtype=np.int32),
            players=np.array([1, -1, -1, -1], dtype=np.int32),
        )
        _, pairs = _encode_trick(rec)
        assert pairs[0, 0] == 5 + CARD_OFFSET
        for i in range(1, NUM_PLAYERS):
            assert pairs[i, 0] == PAD_TOKEN and pairs[i, 1] == PAD_TOKEN

    def test_tokens_in_valid_range(self):
        rec = RoundRecord(
            leading_player=0,
            cards=np.array([0, 51, 25, 13], dtype=np.int32),
            players=np.array([0, 1, 2, 3],  dtype=np.int32),
            winner=0,
        )
        tok, pairs = _encode_trick(rec)
        assert 1 <= tok <= NUM_PLAYERS
        assert (pairs[:, 0] >= 1).all() and (pairs[:, 0] <= NUM_CARDS).all()
        assert (pairs[:, 1] >= 1).all() and (pairs[:, 1] <= NUM_PLAYERS).all()


# ══════════════════════════════════════════════════════════════════════════════
# obs_encoder: PyTreeObs structure
# ══════════════════════════════════════════════════════════════════════════════

class TestPyTreeObsStructure:
    @pytest.fixture
    def obs(self):
        env = SinglePlayerEnv(seed=0)
        o, _ = env.reset()
        return o

    def test_is_named_tuple(self, obs):
        assert isinstance(obs, PyTreeObs)

    def test_no_phase_field(self, obs):
        assert 'phase' not in PyTreeObs._fields

    def test_field_count(self, obs):
        assert len(PyTreeObs._fields) == 11

    def test_field_names(self, obs):
        assert set(PyTreeObs._fields) == set(EXPECTED_FIELDS)

    def test_field_shapes(self, obs):
        for field, (shape, _) in EXPECTED_FIELDS.items():
            assert getattr(obs, field).shape == shape, \
                f"{field}: got {getattr(obs, field).shape}"

    def test_field_dtypes(self, obs):
        for field, (_, dtype) in EXPECTED_FIELDS.items():
            assert getattr(obs, field).dtype == dtype, \
                f"{field}: got {getattr(obs, field).dtype}"

    def test_current_player_in_range(self, obs):
        assert 0 <= obs.current_player[0] < NUM_PLAYERS

    def test_hand_sums_to_13_at_start(self, obs):
        assert obs.hand.sum() == NUM_ROUNDS

    def test_played_all_zero_at_start(self, obs):
        assert obs.played.sum() == 0.0

    def test_scores_zero_at_start(self, obs):
        assert (obs.scores == 0.0).all()

    def test_round_zero_at_start(self, obs):
        assert obs.round[0] == 0.0

    def test_leading_suit_zero_at_start(self, obs):
        assert obs.leading_suit.sum() == 0.0

    def test_hearts_broken_zero_at_start(self, obs):
        assert obs.hearts_broken[0] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# obs_encoder: SinglePlayerEnv correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestSinglePlayerEnv:
    def test_agent_seat_holds_two_of_clubs(self):
        """Agent always starts at the seat holding 2♣."""
        for seed in range(20):
            env = SinglePlayerEnv(seed=seed)
            obs, _ = env.reset()
            seat = int(obs.current_player[0])
            assert env.env.state.hands[seat, TWO_OF_CLUBS], \
                f"seed={seed}: agent seat {seat} doesn't hold 2♣"

    def test_current_player_constant_within_episode(self):
        """Agent's seat (current_player) never changes within an episode."""
        env = SinglePlayerEnv(seed=0)
        obs, _ = env.reset()
        rng  = np.random.default_rng(0)
        seat = int(obs.current_player[0])
        while True:
            mask = env.legal_actions()
            obs, _, term, _, _ = env.step(rng.choice(np.where(mask)[0]))
            assert int(obs.current_player[0]) == seat
            if term:
                break

    def test_exactly_13_agent_steps_per_game(self):
        """Agent acts exactly 13 times per game (once per trick)."""
        for seed in range(10):
            env  = SinglePlayerEnv(seed=seed)
            env.reset()
            rng  = np.random.default_rng(seed)
            steps = 0
            while True:
                mask = env.legal_actions()
                _, _, term, _, _ = env.step(rng.choice(np.where(mask)[0]))
                steps += 1
                if term:
                    break
            assert steps == NUM_ROUNDS, f"seed={seed}: {steps} steps"

    def test_step_called_only_on_agent_turn(self):
        """Underlying env.current_player must equal agent seat at every step() call."""
        for seed in range(10):
            env  = SinglePlayerEnv(seed=seed)
            obs, _ = env.reset()
            rng  = np.random.default_rng(seed)
            seat = int(obs.current_player[0])
            while True:
                assert env.env.state.current_player == seat
                mask = env.legal_actions()
                _, _, term, _, _ = env.step(rng.choice(np.where(mask)[0]))
                if term:
                    break

    def test_legal_actions_shape_and_dtype(self):
        env = SinglePlayerEnv(seed=0)
        env.reset()
        mask = env.legal_actions()
        assert mask.shape == (NUM_CARDS,)
        assert mask.dtype == bool

    def test_legal_actions_never_empty(self):
        env = SinglePlayerEnv(seed=2)
        env.reset()
        rng = np.random.default_rng(2)
        while True:
            mask = env.legal_actions()
            assert mask.any(), "empty legal action mask"
            _, _, term, _, _ = env.step(rng.choice(np.where(mask)[0]))
            if term:
                break

    def test_legal_actions_subset_of_agent_hand(self):
        """Legal cards must be in agent's hand."""
        env  = SinglePlayerEnv(seed=3)
        obs, _ = env.reset()
        rng  = np.random.default_rng(3)
        seat = int(obs.current_player[0])
        while True:
            mask = env.legal_actions()
            hand = env.env.state.hands[seat]
            for c in range(NUM_CARDS):
                if mask[c]:
                    assert hand[c], f"card {c} legal but not in hand"
            _, _, term, _, _ = env.step(rng.choice(np.where(mask)[0]))
            if term:
                break

    def test_reward_zero_until_terminal(self):
        """All non-terminal step rewards are exactly 0.0."""
        for seed in range(10):
            env  = SinglePlayerEnv(seed=seed)
            env.reset()
            rng  = np.random.default_rng(seed)
            while True:
                _, rew, term, _, _ = env.step(rng.choice(np.where(env.legal_actions())[0]))
                if term:
                    break
                assert rew == 0.0, f"seed={seed}: non-terminal step had reward={rew}"

    def test_terminal_reward_equals_terminal_rewards_fn(self):
        """Terminal reward == _terminal_rewards[agent_seat]."""
        for seed in range(10):
            env  = SinglePlayerEnv(seed=seed)
            obs, _ = env.reset()
            rng  = np.random.default_rng(seed)
            seat = int(obs.current_player[0])
            while True:
                _, rew, term, _, _ = env.step(rng.choice(np.where(env.legal_actions())[0]))
                if term:
                    expected = float(env.env._terminal_rewards(env.env.state)[seat])
                    np.testing.assert_allclose(rew, expected, atol=1e-5,
                        err_msg=f"seed={seed}: terminal reward {rew:.4f} != {expected:.4f}")
                    break

    def test_terminal_has_final_scores(self):
        env = SinglePlayerEnv(seed=5)
        steps = _play_single_full(seed=5)
        _, _, term, _, info = steps[-1]
        assert term
        assert "final_scores" in info
        assert info["final_scores"].sum() == MAX_SCORE

    def test_opponent_fn_is_called(self):
        """Opponent callable is invoked during game progression."""
        calls = []
        def tracking_opp(obs, mask):
            calls.append(1)
            return int(np.random.choice(np.where(mask)[0]))

        env = SinglePlayerEnv(opponent_fn=tracking_opp, seed=0)
        env.reset()
        rng = np.random.default_rng(0)
        while True:
            mask = env.legal_actions()
            _, _, term, _, _ = env.step(rng.choice(np.where(mask)[0]))
            if term:
                break
        # 3 opponents × 13 tricks = 39 opponent calls
        assert len(calls) == (NUM_PLAYERS - 1) * NUM_ROUNDS, \
            f"expected {(NUM_PLAYERS-1)*NUM_ROUNDS} opponent calls, got {len(calls)}"

    def test_seat_rotates_across_games(self):
        """Over many resets, agent occupies different seats."""
        env   = SinglePlayerEnv(seed=None)
        seats = set()
        for seed in range(40):
            env.env.rng = np.random.default_rng(seed)
            obs, _ = env.reset()
            seats.add(int(obs.current_player[0]))
        assert len(seats) > 1, "agent always assigned same seat across games"

    def test_reward_accumulates_across_opponent_tricks(self):
        """
        With terminal-only reward: step rewards are 0 mid-game, terminal step
        delivers _terminal_rewards[seat].  Verify over many seeds.
        """
        for seed in range(20):
            env  = SinglePlayerEnv(seed=seed)
            obs, _ = env.reset()
            rng  = np.random.default_rng(seed)
            seat = int(obs.current_player[0])
            step_rewards = []
            while True:
                _, rew, term, _, _ = env.step(rng.choice(np.where(env.legal_actions())[0]))
                step_rewards.append(rew)
                if term:
                    break
            # All intermediate rewards are 0
            assert all(r == 0.0 for r in step_rewards[:-1])
            # Terminal reward matches engine
            expected = float(env.env._terminal_rewards(env.env.state)[seat])
            np.testing.assert_allclose(step_rewards[-1], expected, atol=1e-5)

    def test_obs_encoding_consistency(self):
        """Obs fields reflect game state from agent's perspective."""
        env  = SinglePlayerEnv(seed=6)
        obs, _ = env.reset()
        rng  = np.random.default_rng(6)
        seat = int(obs.current_player[0])
        while True:
            obs2, _, term, _, _ = env.step(rng.choice(np.where(env.legal_actions())[0]))
            s = env.env.state
            # hand should reflect agent's current hand
            np.testing.assert_array_equal(obs2.hand, s.hands[seat].astype(np.float32))
            # scores normalised
            assert (obs2.scores >= 0).all() and (obs2.scores <= 1).all()
            # played cards only grows
            if term:
                break

    def test_played_cards_monotone(self):
        env  = SinglePlayerEnv(seed=7)
        env.reset()
        rng  = np.random.default_rng(7)
        prev = np.zeros(NUM_CARDS, dtype=np.float32)
        while True:
            obs, _, term, _, _ = env.step(rng.choice(np.where(env.legal_actions())[0]))
            assert (obs.played >= prev).all()
            prev = obs.played.copy()
            if term:
                break

    def test_backwards_compat_alias(self):
        """HeartsEnvWrapper is an alias for SinglePlayerEnv."""
        assert HeartsEnvWrapper is SinglePlayerEnv


# ══════════════════════════════════════════════════════════════════════════════
# vec_env
# ══════════════════════════════════════════════════════════════════════════════

class TestVecSinglePlayerEnv:
    def test_reset_shapes(self):
        vec = VecSinglePlayerEnv(num_envs=4)
        obs, masks = vec.reset()
        assert isinstance(obs, PyTreeObs)
        assert masks.shape == (4, NUM_CARDS) and masks.dtype == bool
        for field, (shape, _) in EXPECTED_FIELDS.items():
            assert getattr(obs, field).shape == (4, *shape), field
        vec.close()

    def test_step_shapes(self):
        vec = VecSinglePlayerEnv(num_envs=4)
        obs, masks = vec.reset()
        rng  = np.random.default_rng(0)
        acts = np.array([rng.choice(np.where(m)[0]) for m in masks])
        obs2, masks2, rewards, term, trunc, infos = vec.step(acts)
        assert masks2.shape   == (4, NUM_CARDS)
        assert rewards.shape  == (4,) and rewards.dtype == np.float32
        assert term.shape     == (4,) and trunc.shape == (4,)
        assert (trunc == False).all()
        assert len(infos)     == 4
        vec.close()

    def test_masks_always_non_empty(self):
        vec = VecSinglePlayerEnv(num_envs=4)
        _, masks = vec.reset()
        rng = np.random.default_rng(0)
        for _ in range(300):
            acts = np.array([rng.choice(np.where(m)[0]) for m in masks])
            _, masks, _, _, _, _ = vec.step(acts)
            assert masks.any(axis=1).all()
        vec.close()

    def test_masks_are_52_wide(self):
        """No NO_OP slot — masks are exactly 52 wide."""
        vec = VecSinglePlayerEnv(num_envs=2)
        _, masks = vec.reset()
        assert masks.shape[1] == NUM_CARDS == 52
        vec.close()

    def test_auto_reset_after_terminal(self):
        vec = VecSinglePlayerEnv(num_envs=2, seeds=[0, 1])
        _, masks = vec.reset()
        rng = np.random.default_rng(0)
        seen = [False, False]
        for _ in range(500):
            acts = np.array([rng.choice(np.where(m)[0]) for m in masks])
            obs, masks, _, term, _, infos = vec.step(acts)
            for i, t in enumerate(term):
                if t:
                    seen[i] = True
                    assert "final_scores" in infos[i]
                    assert infos[i]["final_scores"].sum() == MAX_SCORE
            if all(seen):
                break
        assert all(seen)
        vec.close()

    def test_terminal_info_correct(self):
        vec = VecSinglePlayerEnv(num_envs=2, seeds=[10, 11])
        _, masks = vec.reset()
        rng = np.random.default_rng(0)
        found = False
        for _ in range(500):
            acts = np.array([rng.choice(np.where(m)[0]) for m in masks])
            _, masks, _, term, _, infos = vec.step(acts)
            for i, t in enumerate(term):
                if t:
                    assert infos[i]["final_scores"].sum() == MAX_SCORE
                    found = True
            if found:
                break
        assert found
        vec.close()

    def test_reproducibility(self):
        """Same seed produces identical trajectories (agent hand at each step)."""
        def collect(seed, n=25):
            env  = SinglePlayerEnv(seed=seed)
            env.reset()
            rng  = np.random.default_rng(seed ^ 0xDEAD)  # separate action rng
            traj = []
            steps = 0
            while steps < n:
                _, _, term, _, _ = env.step(rng.choice(np.where(env.legal_actions())[0]))
                traj.append(env.env.state.hands[env._seat].copy())
                steps += 1
                if term:
                    env.reset()
            return traj

        for a, b in zip(collect(42), collect(42)):
            np.testing.assert_array_equal(a, b)

    def test_all_fields_have_batch_dim(self):
        vec = VecSinglePlayerEnv(num_envs=5)
        obs, _ = vec.reset()
        for field in PyTreeObs._fields:
            assert getattr(obs, field).shape[0] == 5, field
        vec.close()

    def test_field_dtypes_in_batch(self):
        vec = VecSinglePlayerEnv(num_envs=2)
        obs, _ = vec.reset()
        for f in ["history_leading","history_pairs","current_trick_leading",
                  "current_trick_pairs","current_player"]:
            assert getattr(obs, f).dtype == np.int32, f
        for f in ["scores","round","hand","played","leading_suit","hearts_broken"]:
            assert getattr(obs, f).dtype == np.float32, f
        vec.close()

    def test_custom_opponent_fn(self):
        """Custom opponent_fn is used inside the vec env."""
        calls = []
        def counting_opp(obs, mask):
            calls.append(1)
            return int(np.random.choice(np.where(mask)[0]))

        vec = VecSinglePlayerEnv(num_envs=2, opponent_fn=counting_opp, seeds=[0, 1])
        _, masks = vec.reset()
        rng = np.random.default_rng(0)
        for _ in range(200):
            acts = np.array([rng.choice(np.where(m)[0]) for m in masks])
            _, masks, _, _, _, _ = vec.step(acts)
        assert len(calls) > 0, "opponent_fn never called"
        vec.close()

    def test_backwards_compat_alias(self):
        assert VecHeartsEnv is VecSinglePlayerEnv


# ══════════════════════════════════════════════════════════════════════════════
# Agent  (requires torch)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import torch
    from torch.utils._pytree import tree_map
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _obs_to_tensor(obs: PyTreeObs, device="cpu") -> PyTreeObs:
    def _cvt(x):
        dtype = torch.long if np.issubdtype(x.dtype, np.integer) else torch.float32
        return torch.as_tensor(x, dtype=dtype, device=device)
    return tree_map(_cvt, obs)


def _load_ppo():
    import importlib.util, unittest.mock as mock
    spec = importlib.util.spec_from_file_location(
        "ppo", os.path.join(os.path.dirname(__file__), "ppo.py")
    )
    ppo = importlib.util.module_from_spec(spec)
    with mock.patch.dict("sys.modules", {
        "tyro": mock.MagicMock(),
        "torch.utils.tensorboard": mock.MagicMock(),
    }):
        spec.loader.exec_module(ppo)
    return ppo


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestAgent:
    @pytest.fixture(scope="class")
    def ppo(self):
        return _load_ppo()

    @pytest.fixture
    def batch(self, ppo):
        vec = VecSinglePlayerEnv(num_envs=4, seeds=list(range(4)))
        obs_np, masks_np = vec.reset()
        vec.close()
        obs_t   = _obs_to_tensor(obs_np)
        masks_t = torch.as_tensor(masks_np, dtype=torch.bool)
        agent   = ppo.Agent()
        return agent, obs_t, masks_t

    def test_actor_head_outputs_52(self, ppo):
        """Actor head must output exactly NUM_CARDS=52 logits (no NO_OP)."""
        agent = ppo.Agent()
        assert agent.actor_head.out_features == NUM_CARDS == 52

    def test_no_phase_embedding(self, ppo):
        """Agent must not have a phase_emb attribute."""
        agent = ppo.Agent()
        assert not hasattr(agent, 'phase_emb')

    def test_get_value_shape(self, batch):
        agent, obs_t, _ = batch
        assert agent.get_value(obs_t).shape == (4, 1)

    def test_get_action_and_value_shapes(self, batch):
        agent, obs_t, masks_t = batch
        action, logprob, entropy, value = agent.get_action_and_value(obs_t, masks_t)
        assert action.shape  == (4,)
        assert logprob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape   == (4, 1)

    def test_actions_within_52(self, batch):
        agent, obs_t, masks_t = batch
        for _ in range(20):
            action, _, _, _ = agent.get_action_and_value(obs_t, masks_t)
            assert action.min() >= 0 and action.max() < NUM_CARDS

    def test_sampled_actions_respect_mask(self, batch):
        agent, obs_t, masks_t = batch
        for _ in range(30):
            action, _, _, _ = agent.get_action_and_value(obs_t, masks_t)
            for i, a in enumerate(action):
                assert masks_t[i, a.item()], \
                    f"env {i}: illegal action {a.item()}"

    def test_gradient_flows_to_all_params(self, batch):
        agent, obs_t, masks_t = batch
        _, logprob, entropy, value = agent.get_action_and_value(obs_t, masks_t)
        (-logprob.mean() + value.mean() - entropy.mean()).backward()
        for name, param in agent.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"no gradient: {name}"

    def test_single_env_forward(self, ppo):
        vec = VecSinglePlayerEnv(num_envs=1)
        obs_np, masks_np = vec.reset()
        vec.close()
        agent = ppo.Agent()
        assert agent.get_value(_obs_to_tensor(obs_np)).shape == (1, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestIntegration:
    def test_pytree_storage_roundtrip(self):
        T, N = 8, 3
        vec = VecSinglePlayerEnv(num_envs=N, seeds=list(range(N)))
        obs_np, masks_np = vec.reset()
        obs_t = _obs_to_tensor(obs_np)

        buf = tree_map(
            lambda x: torch.zeros((T, N, *x.shape[1:]), dtype=x.dtype), obs_t
        )
        assert isinstance(buf, PyTreeObs)
        assert 'phase' not in PyTreeObs._fields

        rng = np.random.default_rng(0)
        for step in range(T):
            tree_map(lambda b, v: b.__setitem__(step, v), buf, obs_t)
            acts = np.array([rng.choice(np.where(m)[0]) for m in masks_np])
            obs_np, masks_np, _, _, _, _ = vec.step(acts)
            obs_t = _obs_to_tensor(obs_np)

        b_obs = tree_map(lambda x: x.reshape(-1, *x.shape[2:]), buf)
        assert b_obs.hand.shape          == (T * N, NUM_CARDS)
        assert b_obs.history_pairs.shape == (T * N, 13, 4, 2)

        inds = torch.tensor([0, 2, 5, 8, 11])
        mb = tree_map(lambda x: x[inds], b_obs)
        assert mb.hand.shape == (5, NUM_CARDS)
        vec.close()

    def test_reward_sum_equals_terminal_over_full_episode(self):
        """
        Terminal step reward equals _terminal_rewards[agent_seat].
        All non-terminal step rewards are 0.
        """
        for seed in range(10):
            env  = SinglePlayerEnv(seed=seed)
            obs, _ = env.reset()
            rng  = np.random.default_rng(seed)
            seat = int(obs.current_player[0])
            step_rewards = []
            while True:
                _, rew, term, _, _ = env.step(rng.choice(np.where(env.legal_actions())[0]))
                step_rewards.append(rew)
                if term:
                    break
            assert all(r == 0.0 for r in step_rewards[:-1])
            expected = float(env.env._terminal_rewards(env.env.state)[seat])
            np.testing.assert_allclose(step_rewards[-1], expected, atol=1e-5,
                err_msg=f"seed={seed}: terminal={step_rewards[-1]:.4f} expected={expected:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])