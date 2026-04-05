"""
Tests for R-Öko with abstract SinglePlayerEnv and VecSinglePlayerEnv.

Run with:  python -m pytest test/game/r_eco/test_env.py -v
"""

import numpy as np
import pytest

from abstract import (
    SinglePlayerEnv, VecSinglePlayerEnv, RandomPlayer, key_from_seed, Key,
)
from game.r_eco import (
    RÖkoEngine, RÖkoObs, RÖkoEnvFactory,
    NUM_ACTIONS, NUM_COLORS, NUM_TYPES, STACK_BY_PLAYERS,
    float_dim,
)


def _make_single_env(seed=42, num_players=2) -> SinglePlayerEnv:
    """Create a SinglePlayerEnv with a random opponent."""
    engine_rng, env_rng = key_from_seed(seed).spawn(2)
    engine = RÖkoEngine(rng=engine_rng, num_players=num_players)
    opponent = RandomPlayer()
    return SinglePlayerEnv(engine, opponent.slice(0), rng=env_rng)


class TestSinglePlayerEnv:
    def setup_method(self):
        self.env = _make_single_env(seed=42, num_players=2)

    def test_reset_returns_obs_and_info(self):
        obs, info = self.env.reset()
        assert isinstance(obs, RÖkoObs)
        assert isinstance(info, dict)

    def test_obs_fields_correct_shapes(self):
        obs, _ = self.env.reset()
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
        obs, _ = self.env.reset()
        for field in ('hands', 'recycling_side', 'waste_side',
                      'collected', 'penalty_pile', 'draw_pile_size',
                      'draw_pile_comp'):
            arr = getattr(obs, field)
            assert arr.dtype == np.float32
            assert np.all(arr >= 0.0), f"{field} has negative values"

    def test_legal_actions_mask_shape(self):
        self.env.reset()
        mask = self.env.legal_actions()
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool

    def test_full_episode(self):
        obs, _ = self.env.reset()
        for _ in range(2000):
            mask = self.env.legal_actions()
            action = int(np.random.choice(np.where(mask)[0]))
            obs, reward, terminated, truncated, info = self.env.step(action)
            assert isinstance(obs, RÖkoObs)
            assert np.isfinite(reward)
            if terminated:
                assert "agent_seat" in info
                break
        else:
            pytest.fail("Episode did not terminate within 2000 steps")


class TestVecEnv:
    def setup_method(self):
        key = key_from_seed(42)
        self.opponent = RandomPlayer()
        factory = RÖkoEnvFactory(num_players=2)
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
        assert masks2.shape == (4, NUM_ACTIONS)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)

    def test_auto_reset_on_termination(self):
        obs, masks = self.vec.reset()
        for _ in range(3000):
            actions = np.array([
                int(np.random.choice(np.where(m)[0])) for m in masks
            ])
            obs, masks, rewards, terminated, truncated, infos = self.vec.step(actions)
            for i, done in enumerate(terminated):
                if done:
                    assert "terminal_obs" in infos[i]


class TestResetNeverDone:
    def test_single_env_reset_not_done(self):
        for np_ in [2, 3, 4, 5]:
            for seed in range(200):
                env = _make_single_env(seed=seed, num_players=np_)
                env.reset()
                assert not env.engine.done
                mask = env.legal_actions()
                assert mask.any()

    def test_vec_env_reset_not_done(self):
        for np_ in [2, 3, 4, 5]:
            key = key_from_seed(np_ * 100)
            opp = RandomPlayer()
            factory = RÖkoEnvFactory(num_players=np_)
            vec = VecSinglePlayerEnv(
                num_envs=64, opponent=opp,
                env_factory=factory, key=key,
            )
            obs, masks = vec.reset()
            for i in range(64):
                assert masks[i].any()
            vec.close()

    def test_vec_env_auto_reset_not_done(self):
        key = key_from_seed(999)
        opp = RandomPlayer()
        factory = RÖkoEnvFactory(num_players=3)
        vec = VecSinglePlayerEnv(
            num_envs=32, opponent=opp,
            env_factory=factory, key=key,
        )
        obs, masks = vec.reset()
        for step in range(500):
            actions = np.array([
                np.random.choice(np.where(masks[i])[0])
                for i in range(32)
            ], dtype=np.int32)
            obs, masks, rewards, terminated, truncated, infos = vec.step(actions)
            for i in range(32):
                assert masks[i].any()
        vec.close()
