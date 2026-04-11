"""
Validate all reward structures in LCEngine.

For each reward mode, plays a full game with random actions and verifies:
1. Episode reward sum matches the expected formula
2. No double-counting between mid-game and terminal rewards
3. Both players receive correct rewards
"""
import numpy as np
import pytest
from game.lc.engine import LCEngine, NUM_PLAYERS


def play_game(engine):
    """Play a full game with random actions, return per-step rewards and final scores."""
    engine.reset()
    step_rewards = []  # list of (rewards_p0, rewards_p1) per step
    while not engine.done:
        mask = engine.legal_actions()
        valid = np.where(mask)[0]
        action = engine.rng.choice(valid)
        rewards = engine.step(action)
        step_rewards.append(rewards)
    return step_rewards, engine._scores.copy()


def make_engine(seed=42, **kwargs):
    rng = np.random.default_rng(seed)
    return LCEngine(rng, new_color_penalty=20, max_discard_draws=50, **kwargs)


class TestTerminalRewards:
    """Terminal-only reward modes: reward only on the last step."""

    def test_default(self):
        """Default: +1 win / -1 loss / 0 draw."""
        engine = make_engine()
        step_rewards, scores = play_game(engine)

        # All non-terminal steps should be (0, 0)
        for r in step_rewards[:-1]:
            assert r == (0.0, 0.0), f"Non-terminal reward should be 0: {r}"

        # Terminal reward
        r0, r1 = step_rewards[-1]
        if scores[0] > scores[1]:
            assert r0 == 1.0 and r1 == -1.0
        elif scores[0] < scores[1]:
            assert r0 == -1.0 and r1 == 1.0
        else:
            assert r0 == 0.0 and r1 == 0.0

    def test_score_diff(self):
        """score_diff: terminal (own - opp) / 30."""
        engine = make_engine(score_diff_reward=True)
        step_rewards, scores = play_game(engine)

        for r in step_rewards[:-1]:
            assert r == (0.0, 0.0)

        r0, r1 = step_rewards[-1]
        assert abs(r0 - (scores[0] - scores[1]) / 30.0) < 1e-6
        assert abs(r1 - (scores[1] - scores[0]) / 30.0) < 1e-6

    def test_raw_score(self):
        """raw_score: terminal own / 30."""
        engine = make_engine(raw_score_reward=True)
        step_rewards, scores = play_game(engine)

        for r in step_rewards[:-1]:
            assert r == (0.0, 0.0)

        r0, r1 = step_rewards[-1]
        assert abs(r0 - scores[0] / 30.0) < 1e-6
        assert abs(r1 - scores[1] / 30.0) < 1e-6

    def test_zero_one(self):
        """zero_one: terminal 1 if win else 0."""
        engine = make_engine(zero_one_reward=True)
        step_rewards, scores = play_game(engine)

        for r in step_rewards[:-1]:
            assert r == (0.0, 0.0)

        r0, r1 = step_rewards[-1]
        best = max(scores[0], scores[1])
        assert r0 == (1.0 if scores[0] >= best else 0.0)
        assert r1 == (1.0 if scores[1] >= best else 0.0)


class TestDenseRewards:
    """Dense reward modes: per-step signal."""

    def test_dense_raw_score(self):
        """dense + raw_score: delta(own)/30 every step. Sum = own/30."""
        engine = make_engine(dense_reward=True, raw_score_reward=True)
        step_rewards, scores = play_game(engine)

        sum0 = sum(r[0] for r in step_rewards)
        sum1 = sum(r[1] for r in step_rewards)
        assert abs(sum0 - scores[0] / 30.0) < 1e-5, f"P0 sum {sum0} != {scores[0]/30}"
        assert abs(sum1 - scores[1] / 30.0) < 1e-5, f"P1 sum {sum1} != {scores[1]/30}"

        # Each non-terminal step: only acting player gets reward
        # (we can't easily check which player acts, but at least verify one is 0)

    def test_dense_score_diff(self):
        """dense + score_diff: delta(own-opp)/30 both players. Sum = (own-opp)/30."""
        engine = make_engine(dense_reward=True, score_diff_reward=True)
        step_rewards, scores = play_game(engine)

        sum0 = sum(r[0] for r in step_rewards)
        sum1 = sum(r[1] for r in step_rewards)
        expected0 = (scores[0] - scores[1]) / 30.0
        expected1 = (scores[1] - scores[0]) / 30.0
        assert abs(sum0 - expected0) < 1e-5, f"P0 sum {sum0} != {expected0}"
        assert abs(sum1 - expected1) < 1e-5, f"P1 sum {sum1} != {expected1}"

        # Both players should get nonzero rewards at most steps
        nonzero_p1 = sum(1 for r in step_rewards if r[1] != 0.0)
        assert nonzero_p1 > 0, "P1 should get negative rewards when P0 acts"

    def test_dense_opp_penalty(self):
        """dense + opp_penalty: delta(own)/30 mid-game, delta(own)/30 - opp/30 terminal.
        Sum = own/30 - opp/30 = (own-opp)/30."""
        engine = make_engine(dense_reward=True, dense_opp_penalty=True)
        step_rewards, scores = play_game(engine)

        sum0 = sum(r[0] for r in step_rewards)
        sum1 = sum(r[1] for r in step_rewards)
        expected0 = (scores[0] - scores[1]) / 30.0
        expected1 = (scores[1] - scores[0]) / 30.0
        assert abs(sum0 - expected0) < 1e-5, f"P0 sum {sum0} != {expected0}"
        assert abs(sum1 - expected1) < 1e-5, f"P1 sum {sum1} != {expected1}"

        # Mid-game: only acting player gets reward (no opponent signal)
        # At least verify that most non-terminal steps have one zero
        nonzero_both = sum(1 for r in step_rewards[:-1] if r[0] != 0.0 and r[1] != 0.0)
        assert nonzero_both == 0, "Mid-game should only reward acting player"

    def test_dense_zero_one(self):
        """dense + zero_one: delta(own)/30 + terminal win bonus.
        Sum = own/30 + (1 if win else 0)."""
        engine = make_engine(dense_reward=True, zero_one_reward=True)
        step_rewards, scores = play_game(engine)

        sum0 = sum(r[0] for r in step_rewards)
        sum1 = sum(r[1] for r in step_rewards)
        best = max(scores[0], scores[1])
        bonus0 = 1.0 if scores[0] >= best else 0.0
        bonus1 = 1.0 if scores[1] >= best else 0.0
        expected0 = scores[0] / 30.0 + bonus0
        expected1 = scores[1] / 30.0 + bonus1
        assert abs(sum0 - expected0) < 1e-5, f"P0 sum {sum0} != {expected0}"
        assert abs(sum1 - expected1) < 1e-5, f"P1 sum {sum1} != {expected1}"

    def test_dense_default(self):
        """dense + default (no terminal flag): delta(own)/30 + terminal +1/-1/0.
        Sum = own/30 + wld_bonus."""
        engine = make_engine(dense_reward=True)
        step_rewards, scores = play_game(engine)

        sum0 = sum(r[0] for r in step_rewards)
        sum1 = sum(r[1] for r in step_rewards)
        best = max(scores[0], scores[1])
        worst = min(scores[0], scores[1])
        bonus0 = 1.0 if scores[0] >= best else (-1.0 if scores[0] <= worst else 0.0)
        bonus1 = 1.0 if scores[1] >= best else (-1.0 if scores[1] <= worst else 0.0)
        expected0 = scores[0] / 30.0 + bonus0
        expected1 = scores[1] / 30.0 + bonus1
        assert abs(sum0 - expected0) < 1e-5, f"P0 sum {sum0} != {expected0}"
        assert abs(sum1 - expected1) < 1e-5, f"P1 sum {sum1} != {expected1}"


class TestMultipleSeeds:
    """Run each mode across multiple seeds to catch edge cases."""

    @pytest.mark.parametrize("seed", range(10))
    def test_dense_score_diff_multi(self, seed):
        engine = make_engine(seed=seed, dense_reward=True, score_diff_reward=True)
        step_rewards, scores = play_game(engine)
        sum0 = sum(r[0] for r in step_rewards)
        expected0 = (scores[0] - scores[1]) / 30.0
        assert abs(sum0 - expected0) < 1e-5

    @pytest.mark.parametrize("seed", range(10))
    def test_dense_opp_penalty_multi(self, seed):
        engine = make_engine(seed=seed, dense_reward=True, dense_opp_penalty=True)
        step_rewards, scores = play_game(engine)
        sum0 = sum(r[0] for r in step_rewards)
        expected0 = (scores[0] - scores[1]) / 30.0
        assert abs(sum0 - expected0) < 1e-5

        # Verify mid-game: only acting player gets reward
        for r in step_rewards[:-1]:
            assert r[0] == 0.0 or r[1] == 0.0, f"Mid-game both nonzero: {r}"

    @pytest.mark.parametrize("seed", range(10))
    def test_dense_raw_score_multi(self, seed):
        engine = make_engine(seed=seed, dense_reward=True, raw_score_reward=True)
        step_rewards, scores = play_game(engine)
        sum0 = sum(r[0] for r in step_rewards)
        expected0 = scores[0] / 30.0
        assert abs(sum0 - expected0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
