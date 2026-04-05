"""
Tests for R-Öko game engine (BaseGameEngine implementation).

Run with:  python -m pytest test/game/r_eco/test_engine.py -v
"""

import numpy as np
import pytest

from game.r_eco import (
    RÖkoEngine, RÖkoObs, EcoState,
    NUM_COLORS, NUM_TYPES, NUM_ACTIONS, NUM_PLAY_ACTIONS, NUM_DISCARD_ACTIONS,
    SINGLES_PER_COLOR, DOUBLES_PER_COLOR, HAND_LIMIT, MIN_RECYCLE_VALUE,
    PHASE_PLAY, PHASE_DISCARD, STACK_BY_PLAYERS,
    encode_play, decode_play, encode_discard, decode_discard,
    float_dim,
)


def _make_engine(seed=42, num_players=2) -> RÖkoEngine:
    engine = RÖkoEngine(rng=np.random.default_rng(seed), num_players=num_players)
    engine.reset()
    return engine


# ── Codec tests ──────────────────────────────────────────────────────────────

class TestActionCodec:
    def test_play_roundtrip(self):
        for color in range(NUM_COLORS):
            for n_s in range(5):
                for n_d in range(5):
                    a = encode_play(color, n_s, n_d)
                    assert decode_play(a) == (color, n_s, n_d)

    def test_discard_roundtrip(self):
        for color in range(NUM_COLORS):
            for t in range(NUM_TYPES):
                a = encode_discard(color, t)
                assert decode_discard(a) == (color, t)

    def test_play_range(self):
        for color in range(NUM_COLORS):
            for n_s in range(5):
                for n_d in range(5):
                    a = encode_play(color, n_s, n_d)
                    assert 0 <= a < NUM_PLAY_ACTIONS

    def test_discard_range(self):
        for color in range(NUM_COLORS):
            for t in range(NUM_TYPES):
                a = encode_discard(color, t)
                assert NUM_PLAY_ACTIONS <= a < NUM_ACTIONS

    def test_no_overlap(self):
        play_ids = {encode_play(c, s, d)
                    for c in range(NUM_COLORS)
                    for s in range(5) for d in range(5)}
        discard_ids = {encode_discard(c, t)
                       for c in range(NUM_COLORS) for t in range(NUM_TYPES)}
        assert play_ids.isdisjoint(discard_ids)


# ── Reset tests ──────────────────────────────────────────────────────────────

class TestReset:
    def setup_method(self):
        self.engine = _make_engine(seed=42, num_players=2)
        self.s = self.engine.state

    def test_hands_shape(self):
        assert self.s.hands.shape == (2, NUM_COLORS, NUM_TYPES)

    def test_initial_hand_size(self):
        for p in range(2):
            assert int(self.s.hands[p].sum()) == 3

    def test_initial_phase(self):
        assert self.s.phase == PHASE_PLAY

    def test_recycling_side_empty(self):
        assert self.s.recycling_side.sum() == 0

    def test_waste_side_shape(self):
        assert self.s.waste_side.shape == (NUM_COLORS, NUM_COLORS, NUM_TYPES)

    def test_waste_side_populated(self):
        for f in range(NUM_COLORS):
            assert int(self.s.waste_side[f].sum()) == 1

    def test_factory_stacks_correct(self):
        expected = STACK_BY_PLAYERS[2]
        for c in range(NUM_COLORS):
            assert self.s.factory_stacks[c] == expected

    def test_deck_size(self):
        total = NUM_COLORS * (SINGLES_PER_COLOR + DOUBLES_PER_COLOR)
        # dealt: 2 players x 3 + 4 factories x 1 = 10 cards removed
        assert len(self.s.draw_pile) == total - 10

    def test_penalty_pile_empty(self):
        assert self.s.penalty_pile.sum() == 0


# ── Legal actions tests ──────────────────────────────────────────────────────

class TestLegalActions:
    def setup_method(self):
        self.engine = _make_engine(seed=42, num_players=2)

    def test_legal_mask_shape(self):
        mask = self.engine.legal_actions()
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool

    def test_only_play_actions_in_play_phase(self):
        mask = self.engine.legal_actions()
        assert not mask[NUM_PLAY_ACTIONS:].any()
        assert mask[:NUM_PLAY_ACTIONS].any()

    def test_no_zero_card_play(self):
        mask = self.engine.legal_actions()
        for color in range(NUM_COLORS):
            assert not mask[encode_play(color, 0, 0)]

    def test_cannot_play_more_than_hand(self):
        s = self.engine.state
        p = s.current_player
        mask = self.engine.legal_actions()
        for color in range(NUM_COLORS):
            max_s = int(s.hands[p, color, 0])
            max_d = int(s.hands[p, color, 1])
            for n_s in range(5):
                for n_d in range(5):
                    a = encode_play(color, n_s, n_d)
                    if n_s > max_s or n_d > max_d:
                        assert not mask[a]


# ── Step tests ───────────────────────────────────────────────────────────────

class TestStep:
    def setup_method(self):
        self.engine = _make_engine(seed=7, num_players=2)

    def test_illegal_action_raises(self):
        with pytest.raises(AssertionError):
            self.engine.step(encode_play(0, 0, 0))

    def test_step_removes_cards_from_hand(self):
        s = self.engine.state
        p = s.current_player
        mask = self.engine.legal_actions()
        action = int(np.where(mask)[0][0])
        color, n_s, n_d = decode_play(action)
        waste_s = int(s.waste_side[color, color, 0])
        waste_d = int(s.waste_side[color, color, 1])
        hand_before = s.hands[p, color].copy()
        self.engine.step(action)
        assert s.hands[p, color, 0] == hand_before[0] - n_s + waste_s
        assert s.hands[p, color, 1] == hand_before[1] - n_d + waste_d

    def test_step_adds_to_recycling_side(self):
        s = self.engine.state
        mask = self.engine.legal_actions()
        action = int(np.where(mask)[0][0])
        color, n_s, n_d = decode_play(action)
        rec_before = s.recycling_side[color].copy()
        self.engine.step(action)
        # Either recycling was cleared (if >= MIN_RECYCLE_VALUE) or incremented
        assert (s.recycling_side[color, 0] == rec_before[0] + n_s
                or s.recycling_side[color, 0] == 0)

    def test_turn_advances_to_next_player(self):
        s = self.engine.state
        p0 = s.current_player
        mask = self.engine.legal_actions()
        action = int(np.where(mask)[0][0])
        self.engine.step(action)
        if s.phase == PHASE_PLAY:
            assert s.current_player != p0 or s.done

    def test_rewards_type(self):
        mask = self.engine.legal_actions()
        action = int(np.where(mask)[0][0])
        rewards = self.engine.step(action)
        assert isinstance(rewards, tuple)
        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)


# ── Discard tests ────────────────────────────────────────────────────────────

class TestDiscard:
    def test_discard_reduces_hand(self):
        engine = _make_engine(seed=99, num_players=2)
        s = engine.state
        p = s.current_player
        s.hands[p, 0, 0] = 4
        s.hands[p, 1, 0] = 3
        s.phase = PHASE_DISCARD

        mask = engine.legal_actions()
        assert mask[NUM_PLAY_ACTIONS:].any()
        action = int(np.where(mask)[0][0])
        hand_before = int(s.hands[p].sum())
        engine.step(action)
        assert int(s.hands[p].sum()) == hand_before - 1

    def test_discard_adds_to_penalty_pile(self):
        engine = _make_engine(seed=99, num_players=2)
        s = engine.state
        p = s.current_player
        s.hands[p, 0, 0] = 4
        s.hands[p, 1, 0] = 3
        s.phase = PHASE_DISCARD
        pen_before = s.penalty_pile[p].sum()
        mask = engine.legal_actions()
        action = int(np.where(mask)[0][0])
        color, t = decode_discard(action)
        engine.step(action)
        assert s.penalty_pile[p].sum() == pen_before + 1
        assert s.penalty_pile[p, color, t] == 1


# ── Scoring tests ────────────────────────────────────────────────────────────

class TestScoring:
    def test_scoring_requires_more_than_one_card(self):
        engine = _make_engine(seed=0, num_players=2)
        s = engine.state
        s.collected[0][0] = [5]
        s.collected[0][1] = [4, 1]
        scores = engine.compute_scores()
        assert scores[0] == 5.0

    def test_penalty_deduction(self):
        engine = _make_engine(seed=0, num_players=2)
        s = engine.state
        s.collected[0][0] = [1, 2]
        s.penalty_pile[0, 0, 0] = 2
        scores = engine.compute_scores()
        assert scores[0] == 1.0

    def test_clean_player_bonus_2p(self):
        engine = _make_engine(seed=0, num_players=2)
        s = engine.state
        s.penalty_pile[1, 0, 0] = 1
        scores = engine.compute_scores()
        assert scores[0] == 3.0

    def test_no_bonus_if_all_clean(self):
        engine = _make_engine(seed=0, num_players=2)
        scores = engine.compute_scores()
        assert scores[0] == 0.0
        assert scores[1] == 0.0

    def test_negative_factory_card(self):
        engine = _make_engine(seed=0, num_players=2)
        s = engine.state
        s.collected[0][0] = [-2, 1]
        scores = engine.compute_scores()
        assert scores[0] == -1.0


# ── Encode tests ─────────────────────────────────────────────────────────────

class TestEncode:
    def test_obs_type(self):
        engine = _make_engine(seed=42, num_players=2)
        obs = engine.encode(0)
        assert isinstance(obs, RÖkoObs)

    def test_obs_shapes(self):
        engine = _make_engine(seed=42, num_players=2)
        obs = engine.encode(0)
        num_p = 2
        assert obs.current_player.shape == (1,)
        assert obs.phase.shape == (1,)
        assert obs.hands.shape == (num_p * NUM_COLORS * NUM_TYPES,)
        assert obs.recycling_side.shape == (NUM_COLORS * NUM_TYPES,)
        assert obs.waste_side.shape == (NUM_COLORS * NUM_COLORS * NUM_TYPES,)
        stack_size = len(STACK_BY_PLAYERS[2])
        assert obs.factory_stacks.shape == (NUM_COLORS * stack_size,)
        assert obs.collected.shape == (num_p * NUM_COLORS * 2,)
        assert obs.penalty_pile.shape == (num_p * NUM_COLORS * NUM_TYPES,)
        assert obs.scores.shape == (num_p,)
        assert obs.draw_pile_size.shape == (1,)
        assert obs.draw_pile_comp.shape == (NUM_COLORS * NUM_TYPES,)

    def test_float_fields_in_range(self):
        engine = _make_engine(seed=42, num_players=2)
        obs = engine.encode(0)
        for field in ('hands', 'recycling_side', 'waste_side',
                      'collected', 'penalty_pile', 'draw_pile_size',
                      'draw_pile_comp'):
            arr = getattr(obs, field)
            assert arr.dtype == np.float32
            assert np.all(arr >= 0.0), f"{field} has negative values"
        fs = obs.factory_stacks
        assert fs.dtype == np.float32
        assert np.all(fs >= -1.0) and np.all(fs <= 1.0)

    def test_int_fields_dtype(self):
        engine = _make_engine(seed=42, num_players=2)
        obs = engine.encode(0)
        assert obs.current_player.dtype == np.int32
        assert obs.phase.dtype == np.int32

    def test_relative_seat_always_zero(self):
        engine = _make_engine(seed=42, num_players=2)
        obs = engine.encode(0)
        assert obs.current_player[0] == 0
        obs1 = engine.encode(1)
        assert obs1.current_player[0] == 0

    def test_float_dim(self):
        stack_size = len(STACK_BY_PLAYERS[2])
        assert float_dim(2) == (
            2 * NUM_COLORS * NUM_TYPES
            + NUM_COLORS * NUM_TYPES
            + NUM_COLORS * NUM_COLORS * NUM_TYPES
            + NUM_COLORS * stack_size
            + 2 * NUM_COLORS * 2
            + 2 * NUM_COLORS * NUM_TYPES
            + 2
            + 1
            + NUM_COLORS * NUM_TYPES
        )


# ── Full game tests ──────────────────────────────────────────────────────────

class TestFullGame:
    def test_game_terminates(self):
        engine = _make_engine(seed=42, num_players=2)
        for _ in range(5000):
            if engine.done:
                break
            mask = engine.legal_actions()
            action = int(np.random.choice(np.where(mask)[0]))
            engine.step(action)
        assert engine.done

    def test_legal_actions_always_nonempty(self):
        engine = _make_engine(seed=55, num_players=2)
        for _ in range(500):
            if engine.done:
                break
            mask = engine.legal_actions()
            assert mask.any()
            action = int(np.random.choice(np.where(mask)[0]))
            engine.step(action)

    def test_hand_limit_never_exceeded(self):
        engine = _make_engine(seed=77, num_players=2)
        for _ in range(500):
            if engine.done:
                break
            mask = engine.legal_actions()
            action = int(np.random.choice(np.where(mask)[0]))
            engine.step(action)
            s = engine.state
            if s.phase == PHASE_PLAY:
                for p in range(engine.num_players):
                    assert int(s.hands[p].sum()) <= HAND_LIMIT

    def test_rewards_finite(self):
        engine = _make_engine(seed=11, num_players=2)
        for _ in range(1000):
            if engine.done:
                break
            mask = engine.legal_actions()
            action = int(np.random.choice(np.where(mask)[0]))
            rewards = engine.step(action)
            assert all(np.isfinite(r) for r in rewards)

    def test_terminal_rewards_emitted(self):
        """On game end, step() should include large +1/-1 terminal signal."""
        engine = _make_engine(seed=42, num_players=2)
        terminal_rewards = None
        for _ in range(5000):
            if engine.done:
                break
            mask = engine.legal_actions()
            action = int(np.random.choice(np.where(mask)[0]))
            rewards = engine.step(action)
            if engine.done:
                terminal_rewards = rewards
        assert terminal_rewards is not None
        # Terminal rewards dominate: at least one player > 0.5 (win)
        assert max(terminal_rewards) > 0.5
        # At least one player < -0.5 (loss) or all tie
        assert min(terminal_rewards) < -0.5 or all(r > 0.5 for r in terminal_rewards)

    @pytest.mark.parametrize("num_players", [2, 3, 4, 5])
    def test_all_player_counts(self, num_players):
        engine = RÖkoEngine(rng=np.random.default_rng(42), num_players=num_players)
        engine.reset()
        for _ in range(5000):
            if engine.done:
                break
            mask = engine.legal_actions()
            action = int(np.random.choice(np.where(mask)[0]))
            engine.step(action)
        assert engine.done
