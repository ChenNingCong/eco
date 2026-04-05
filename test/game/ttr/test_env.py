"""
Tests for Ticket to Ride with abstract SinglePlayerEnv and VecSinglePlayerEnv.

Run with:  python -m pytest test/game/ttr/test_env.py -v
"""

import numpy as np
import pytest

from abstract import (
    SinglePlayerEnv, VecSinglePlayerEnv, RandomPlayer, key_from_seed,
)
from game.ttr import (
    TTREngine, TTRObs, TTREnvFactory, NUM_ACTIONS,
    NUM_ROUTES, NUM_DESTINATIONS, GAME_OVER,
)
from game.ttr.engine import float_dim


def _make_single_env(seed=42, num_players=2) -> SinglePlayerEnv:
    engine_rng, env_rng = key_from_seed(seed).spawn(2)
    engine = TTREngine(rng=engine_rng, num_players=num_players)
    opponent = RandomPlayer()
    return SinglePlayerEnv(engine, opponent.slice(0), rng=env_rng)


class TestEngine:
    def test_reset_and_encode(self):
        engine = TTREngine(rng=np.random.default_rng(42))
        engine.reset()
        obs = engine.encode(0)
        assert isinstance(obs, TTRObs)
        assert obs.game_state.shape == (1,)
        assert obs.hands.shape == (9 * 2,)  # 2 players * 9 colors
        assert obs.route_ownership.shape == (NUM_ROUTES * 2,)
        assert obs.own_dest_status.shape == (NUM_DESTINATIONS,)

    def test_legal_actions_nonempty_after_reset(self):
        engine = TTREngine(rng=np.random.default_rng(42))
        engine.reset()
        mask = engine.legal_actions()
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.any()

    def test_first_round_requires_draw_dest(self):
        """In FIRST_ROUND, only DrawDestinations should be legal (from INIT)."""
        engine = TTREngine(rng=np.random.default_rng(42))
        engine.reset()
        mask = engine.legal_actions()
        # Only action 10 (DrawDest) should be true in INIT during FIRST_ROUND
        assert mask[10]  # DrawDestinations
        assert not mask[0]  # DrawRandom should NOT be legal in first round

    def test_full_random_game(self):
        """Play a full game with random actions."""
        rng = np.random.default_rng(42)
        engine = TTREngine(rng=rng)
        engine.reset()
        for _ in range(5000):
            if engine.done:
                break
            mask = engine.legal_actions()
            assert mask.any(), "No legal actions but game not done"
            action = int(rng.choice(np.where(mask)[0]))
            engine.step(action)
        assert engine.done, "Game did not terminate within 5000 steps"

    def test_multiple_seeds(self):
        """Multiple seeds all produce valid complete games."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            engine = TTREngine(rng=rng)
            engine.reset()
            steps = 0
            while not engine.done and steps < 5000:
                mask = engine.legal_actions()
                assert mask.any()
                action = int(rng.choice(np.where(mask)[0]))
                engine.step(action)
                steps += 1
            assert engine.done, f"Seed {seed}: game did not terminate in {steps} steps"

    @pytest.mark.parametrize("num_players", [2, 3, 4, 5])
    def test_all_player_counts(self, num_players):
        rng = np.random.default_rng(42)
        engine = TTREngine(rng=rng, num_players=num_players)
        engine.reset()
        for _ in range(10000):
            if engine.done:
                break
            mask = engine.legal_actions()
            assert mask.any()
            engine.step(int(rng.choice(np.where(mask)[0])))
        assert engine.done

    def test_scores_finite(self):
        rng = np.random.default_rng(99)
        engine = TTREngine(rng=rng)
        engine.reset()
        while not engine.done:
            mask = engine.legal_actions()
            action = int(rng.choice(np.where(mask)[0]))
            engine.step(action)
        scores = engine.compute_scores()
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))

    def test_float_dim(self):
        assert float_dim(2) == 293


class TestSinglePlayerEnv:
    def setup_method(self):
        self.env = _make_single_env(seed=42)

    def test_reset_returns_obs_and_info(self):
        obs, info = self.env.reset()
        assert isinstance(obs, TTRObs)
        assert isinstance(info, dict)

    def test_full_episode(self):
        obs, _ = self.env.reset()
        for _ in range(5000):
            mask = self.env.legal_actions()
            action = int(np.random.choice(np.where(mask)[0]))
            obs, reward, terminated, truncated, info = self.env.step(action)
            assert isinstance(obs, TTRObs)
            assert np.isfinite(reward)
            if terminated:
                assert "agent_seat" in info
                break
        else:
            pytest.fail("Episode did not terminate within 5000 steps")


class TestVecEnv:
    def setup_method(self):
        key = key_from_seed(42)
        self.opponent = RandomPlayer()
        factory = TTREnvFactory(num_players=2)
        self.vec = VecSinglePlayerEnv(
            num_envs=4, opponent=self.opponent,
            env_factory=factory, key=key,
        )

    def test_reset_shapes(self):
        obs, masks = self.vec.reset()
        assert obs.hands.shape[0] == 4
        assert masks.shape == (4, NUM_ACTIONS)

    def test_step_output_shapes(self):
        obs, masks = self.vec.reset()
        actions = np.array([int(np.random.choice(np.where(m)[0])) for m in masks])
        obs2, masks2, rewards, terminated, truncated, infos = self.vec.step(actions)
        assert obs2.hands.shape[0] == 4
        assert rewards.shape == (4,)

    def test_auto_reset_on_termination(self):
        obs, masks = self.vec.reset()
        for _ in range(10000):
            actions = np.array([
                int(np.random.choice(np.where(m)[0])) for m in masks
            ])
            obs, masks, rewards, terminated, truncated, infos = self.vec.step(actions)
            for i, done in enumerate(terminated):
                if done:
                    assert "terminal_obs" in infos[i]
                    # After auto-reset, mask should be valid
                    assert masks[i].any()
