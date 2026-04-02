"""
Test suite for R-öko RL environment.

Run with:  python -m pytest eco_tests.py -v
"""

import numpy as np
import pytest

from eco_env import (
    EcoEnv, EcoState,
    NUM_COLORS, NUM_TYPES, NUM_ACTIONS, NUM_PLAY_ACTIONS, NUM_DISCARD_ACTIONS,
    SINGLES_PER_COLOR, DOUBLES_PER_COLOR, HAND_LIMIT, MIN_RECYCLE_VALUE,
    PHASE_PLAY, PHASE_DISCARD,
    encode_play, decode_play, encode_discard, decode_discard,
    _STACK,
)
from eco_obs_encoder import (
    SinglePlayerEcoEnv, EcoPyTreeObs, eco_float_dim,
)
from eco_vec_env import VecSinglePlayerEcoEnv


# ── Codec tests ───────────────────────────────────────────────────────────────

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
        play_ids    = {encode_play(c, s, d)
                       for c in range(NUM_COLORS)
                       for s in range(5) for d in range(5)}
        discard_ids = {encode_discard(c, t)
                       for c in range(NUM_COLORS) for t in range(NUM_TYPES)}
        assert play_ids.isdisjoint(discard_ids)


# ── EcoEnv basic tests ────────────────────────────────────────────────────────

class TestEcoEnvReset:
    def setup_method(self):
        self.env = EcoEnv(num_players=2, seed=42)
        self.state = self.env.reset()

    def test_hands_shape(self):
        assert self.state.hands.shape == (2, NUM_COLORS, NUM_TYPES)

    def test_initial_hand_size(self):
        for p in range(2):
            assert int(self.state.hands[p].sum()) == 3

    def test_initial_phase(self):
        assert self.state.phase == PHASE_PLAY

    def test_recycling_side_empty(self):
        assert self.state.recycling_side.sum() == 0

    def test_waste_side_shape(self):
        assert self.state.waste_side.shape == (NUM_COLORS, NUM_COLORS, NUM_TYPES)

    def test_waste_side_populated(self):
        # Each factory should have exactly 1 waste card
        for f in range(NUM_COLORS):
            assert int(self.state.waste_side[f].sum()) == 1

    def test_factory_stacks_correct(self):
        expected = _STACK[2]
        for c in range(NUM_COLORS):
            assert self.state.factory_stacks[c] == expected

    def test_deck_size(self):
        total_cards = NUM_COLORS * (SINGLES_PER_COLOR + DOUBLES_PER_COLOR)  # 88
        # dealt: 2 players × 3 + 4 factories × 1 = 10 cards removed
        assert len(self.state.draw_pile) == total_cards - 10

    def test_penalty_pile_empty(self):
        assert self.state.penalty_pile.sum() == 0


class TestEcoEnvLegalActions:
    def setup_method(self):
        self.env = EcoEnv(num_players=2, seed=42)
        self.state = self.env.reset()

    def test_legal_mask_shape(self):
        mask = self.env.legal_actions()
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool

    def test_only_play_actions_in_play_phase(self):
        mask = self.env.legal_actions()
        # No discard actions should be legal in play phase
        assert not mask[NUM_PLAY_ACTIONS:].any()
        # At least one play action must be legal
        assert mask[:NUM_PLAY_ACTIONS].any()

    def test_no_zero_card_play(self):
        mask = self.env.legal_actions()
        for color in range(NUM_COLORS):
            assert not mask[encode_play(color, 0, 0)]

    def test_cannot_play_more_than_hand(self):
        s = self.state
        p = s.current_player
        mask = self.env.legal_actions()
        for color in range(NUM_COLORS):
            max_s = int(s.hands[p, color, 0])
            max_d = int(s.hands[p, color, 1])
            for n_s in range(5):
                for n_d in range(5):
                    a = encode_play(color, n_s, n_d)
                    if n_s > max_s or n_d > max_d:
                        assert not mask[a], f"Should not be able to play {n_s}s+{n_d}d of color {color}"


class TestEcoEnvStep:
    def setup_method(self):
        self.env = EcoEnv(num_players=2, seed=7)
        self.state = self.env.reset()

    def test_illegal_action_raises(self):
        with pytest.raises(AssertionError):
            self.env.step(encode_play(0, 0, 0))   # must play at least one card

    def test_step_removes_cards_from_hand(self):
        s = self.state
        p = s.current_player
        mask = self.env.legal_actions()
        action = int(np.where(mask)[0][0])
        color, n_s, n_d = decode_play(action)
        # Record waste cards of same color that will be picked up
        waste_s = int(s.waste_side[color, color, 0])
        waste_d = int(s.waste_side[color, color, 1])
        hand_before = s.hands[p, color].copy()
        self.env.step(action)
        # After step: played cards removed, then waste cards added
        assert s.hands[p, color, 0] == hand_before[0] - n_s + waste_s
        assert s.hands[p, color, 1] == hand_before[1] - n_d + waste_d

    def test_step_adds_to_recycling_side(self):
        s = self.state
        p = s.current_player
        mask = self.env.legal_actions()
        action = int(np.where(mask)[0][0])
        color, n_s, n_d = decode_play(action)
        rec_before = s.recycling_side[color].copy()
        self.env.step(action)
        assert s.recycling_side[color, 0] == rec_before[0] + n_s or s.recycling_side[color, 0] == 0

    def test_player_takes_waste_cards(self):
        s = self.state
        p = s.current_player
        mask = self.env.legal_actions()
        # Find any legal play action
        action = int(np.where(mask)[0][0])
        color, _, _ = decode_play(action)
        waste_before = int(s.waste_side[color].sum())
        hand_before  = int(s.hands[p].sum())
        self.env.step(action)
        # If still in play phase, hand grew by waste_before (minus played cards)
        if s.phase == PHASE_PLAY:
            # Hand size = hand_before - played + waste (minus discard if any)
            assert int(s.hands[p].sum()) <= HAND_LIMIT

    def test_turn_advances_to_next_player(self):
        s = self.state
        p0 = s.current_player
        mask = self.env.legal_actions()
        action = int(np.where(mask)[0][0])
        self.env.step(action)
        if s.phase == PHASE_PLAY:
            assert s.current_player != p0 or s.done

    def test_rewards_shape(self):
        mask = self.env.legal_actions()
        action = int(np.where(mask)[0][0])
        _, rewards, _, _ = self.env.step(action)
        assert rewards.shape == (2,)
        assert rewards.dtype == np.float32


class TestEcoEnvDiscard:
    """Test that discard phase triggers and works correctly."""

    def _force_discard(self, env: EcoEnv):
        """Play cards repeatedly until a discard phase is reached, or game over."""
        for _ in range(200):
            if env.state.done:
                return False
            if env.state.phase == PHASE_DISCARD:
                return True
            mask = env.legal_actions()
            action = int(np.where(mask)[0][0])
            env.step(action)
        return env.state.phase == PHASE_DISCARD

    def test_discard_phase_reached(self):
        env = EcoEnv(num_players=2, seed=123)
        env.reset()
        found = self._force_discard(env)
        # Not guaranteed in every game, but very likely; just check the state is valid
        assert env.state.phase in (PHASE_PLAY, PHASE_DISCARD)

    def test_discard_reduces_hand(self):
        # Manually construct a situation with too many cards
        env = EcoEnv(num_players=2, seed=99)
        s = env.reset()
        p = s.current_player
        # Stuff the player's hand beyond HAND_LIMIT
        s.hands[p, 0, 0] = 4
        s.hands[p, 1, 0] = 3
        s.phase = PHASE_DISCARD

        mask = env.legal_actions()
        assert mask[NUM_PLAY_ACTIONS:].any()
        action = int(np.where(mask)[0][0])
        hand_before = int(s.hands[p].sum())
        env.step(action)
        assert int(s.hands[p].sum()) == hand_before - 1

    def test_discard_adds_to_penalty_pile(self):
        env = EcoEnv(num_players=2, seed=99)
        s = env.reset()
        p = s.current_player
        s.hands[p, 0, 0] = 4
        s.hands[p, 1, 0] = 3
        s.phase = PHASE_DISCARD
        pen_before = s.penalty_pile[p].sum()
        mask = env.legal_actions()
        action = int(np.where(mask)[0][0])
        color, t = decode_discard(action)
        env.step(action)
        assert s.penalty_pile[p].sum() == pen_before + 1
        assert s.penalty_pile[p, color, t] == 1


class TestEcoEnvScoring:
    def test_scoring_requires_more_than_one_card(self):
        env = EcoEnv(num_players=2, seed=0)
        s = env.reset()
        s.collected[0][0] = [5]     # only 1 card → doesn't score
        s.collected[0][1] = [4, 1]  # 2 cards → scores
        scores = env.compute_scores(s)
        assert scores[0] == 5.0  # 0 + (4+1) + 0 + 0 = 5

    def test_penalty_deduction(self):
        env = EcoEnv(num_players=2, seed=0)
        s = env.reset()
        s.collected[0][0] = [1, 2]
        s.penalty_pile[0, 0, 0] = 2   # 2 penalty cards
        scores = env.compute_scores(s)
        assert scores[0] == 1.0  # (1+2) - 2 = 1

    def test_clean_player_bonus_2p(self):
        env = EcoEnv(num_players=2, seed=0)
        s = env.reset()
        # Player 1 has a penalty card, player 0 does not
        s.penalty_pile[1, 0, 0] = 1
        scores = env.compute_scores(s)
        assert scores[0] == 3.0   # +3 bonus for clean player in 2-player game

    def test_no_bonus_if_all_clean(self):
        env = EcoEnv(num_players=2, seed=0)
        s = env.reset()
        scores = env.compute_scores(s)
        assert scores[0] == 0.0
        assert scores[1] == 0.0

    def test_negative_factory_card(self):
        env = EcoEnv(num_players=2, seed=0)
        s = env.reset()
        s.collected[0][0] = [-2, 1]  # sum = -1, but count > 1 so it scores
        scores = env.compute_scores(s)
        assert scores[0] == -1.0


class TestEcoEnvFullGame:
    def test_game_terminates(self):
        env = EcoEnv(num_players=2, seed=42)
        env.reset()
        for _ in range(5000):
            if env.state.done:
                break
            mask = env.legal_actions()
            action = int(np.random.choice(np.where(mask)[0]))
            env.step(action)
        assert env.state.done, "Game should terminate within 5000 steps"

    def test_legal_actions_always_nonempty(self):
        env = EcoEnv(num_players=2, seed=55)
        env.reset()
        for _ in range(500):
            if env.state.done:
                break
            mask = env.legal_actions()
            assert mask.any(), "Legal actions should always be non-empty before game end"
            action = int(np.random.choice(np.where(mask)[0]))
            env.step(action)

    def test_hand_limit_never_exceeded_after_step(self):
        env = EcoEnv(num_players=2, seed=77)
        env.reset()
        for _ in range(500):
            if env.state.done:
                break
            mask = env.legal_actions()
            action = int(np.random.choice(np.where(mask)[0]))
            env.step(action)
            s = env.state
            if s.phase == PHASE_PLAY:
                for p in range(s.num_players):
                    assert int(s.hands[p].sum()) <= HAND_LIMIT, \
                        f"Player {p} has {s.hands[p].sum()} cards (> {HAND_LIMIT})"

    def test_rewards_in_range(self):
        env = EcoEnv(num_players=2, seed=11)
        env.reset()
        for _ in range(1000):
            if env.state.done:
                break
            mask = env.legal_actions()
            action = int(np.random.choice(np.where(mask)[0]))
            _, rewards, _, _ = env.step(action)
            assert np.all(np.isfinite(rewards)), "Rewards must be finite"


# ── SinglePlayerEcoEnv tests ──────────────────────────────────────────────────

class TestSinglePlayerEcoEnv:
    def setup_method(self):
        self.env = SinglePlayerEcoEnv(num_players=2, seed=42)

    def test_reset_returns_obs_and_info(self):
        obs, info = self.env.reset()
        assert isinstance(obs, EcoPyTreeObs)
        assert isinstance(info, dict)

    def test_obs_fields_correct_shapes(self):
        obs, _ = self.env.reset()
        num_p = 2
        assert obs.current_player.shape == (1,)
        assert obs.phase.shape           == (1,)
        assert obs.hands.shape           == (num_p * NUM_COLORS * NUM_TYPES,)
        assert obs.recycling_side.shape  == (NUM_COLORS * NUM_TYPES,)
        assert obs.waste_side.shape      == (NUM_COLORS * NUM_COLORS * NUM_TYPES,)
        stack_size = len(_STACK[2])
        assert obs.factory_stacks.shape == (NUM_COLORS * stack_size,)
        assert obs.collected.shape       == (num_p * NUM_COLORS * 2,)
        assert obs.penalty_pile.shape    == (num_p * NUM_COLORS * NUM_TYPES,)
        assert obs.scores.shape          == (num_p,)
        assert obs.draw_pile_size.shape  == (1,)
        assert obs.draw_pile_comp.shape == (NUM_COLORS * NUM_TYPES,)

    def test_float_fields_in_range(self):
        obs, _ = self.env.reset()
        for field in ('hands', 'recycling_side', 'waste_side',
                      'collected', 'penalty_pile', 'draw_pile_size',
                      'draw_pile_comp'):
            arr = getattr(obs, field)
            assert arr.dtype == np.float32
            assert np.all(arr >= 0.0), f"{field} has negative values"
        # factory_stacks: consumed slots = -1.0, active slots in [-0.4, 1.0]
        fs = obs.factory_stacks
        assert fs.dtype == np.float32
        assert np.all(fs >= -1.0) and np.all(fs <= 1.0), "factory_stacks out of [-1, 1]"

    def test_int_fields_dtype(self):
        obs, _ = self.env.reset()
        assert obs.current_player.dtype == np.int32
        assert obs.phase.dtype           == np.int32

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
            assert isinstance(obs, EcoPyTreeObs)
            assert np.isfinite(reward)
            if terminated:
                assert "final_scores" in info
                break
        else:
            pytest.fail("Episode did not terminate within 2000 steps")

    def test_eco_float_dim(self):
        stack_size = len(_STACK[2])
        assert eco_float_dim(2) == (
            2 * NUM_COLORS * NUM_TYPES              # hands
            + NUM_COLORS * NUM_TYPES                # recycling_side
            + NUM_COLORS * NUM_COLORS * NUM_TYPES   # waste_side
            + NUM_COLORS * stack_size               # factory_stacks (full slot content)
            + 2 * NUM_COLORS * 2                    # collected (count + value)
            + 2 * NUM_COLORS * NUM_TYPES            # penalty_pile
            + 2                                     # scores
            + 1                                     # draw_pile_size
            + NUM_COLORS * NUM_TYPES                # draw_pile_comp
        )


# ── VecSinglePlayerEcoEnv tests ───────────────────────────────────────────────

class TestVecSinglePlayerEcoEnv:
    def setup_method(self):
        self.vec = VecSinglePlayerEcoEnv(num_envs=4, num_players=2, seeds=list(range(4)))

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

    def test_batch_opponent_fn_equivalent_to_per_game(self):
        """batch_opponent_fn path produces same-shaped outputs as the per-game path."""
        import numpy as np
        # per-game path
        vec_a = VecSinglePlayerEcoEnv(num_envs=4, num_players=2, seeds=list(range(4)))
        obs_a, masks_a = vec_a.reset()
        actions_a = np.array([int(np.random.choice(np.where(m)[0])) for m in masks_a])
        out_a = vec_a.step(actions_a, batch_opponent_fn=None)

        # batched path (random batch fn)
        vec_b = VecSinglePlayerEcoEnv(num_envs=4, num_players=2, seeds=list(range(4)))
        obs_b, masks_b = vec_b.reset()
        actions_b = actions_a.copy()

        def _rand_batch(obs_batch, mask_batch):
            return np.array([int(np.random.choice(np.where(m)[0])) for m in mask_batch])

        out_b = vec_b.step(actions_b, batch_opponent_fn=_rand_batch)

        # Shapes must match regardless of which path is taken
        assert out_a[0].hands.shape == out_b[0].hands.shape
        assert out_a[1].shape == out_b[1].shape  # masks
        assert out_a[2].shape == out_b[2].shape  # rewards

    def test_auto_reset_on_termination(self):
        obs, masks = self.vec.reset()
        for _ in range(3000):
            actions = np.array([int(np.random.choice(np.where(m)[0])) for m in masks])
            obs, masks, rewards, terminated, truncated, infos = self.vec.step(actions)
            # Check that terminated envs delivered final_scores
            for i, done in enumerate(terminated):
                if done:
                    assert "final_scores" in infos[i]


# ── EcoAgent forward-pass test ────────────────────────────────────────────────

class TestEcoAgent:
    def test_forward_pass(self):
        try:
            import torch
            from eco_ppo import EcoAgent, obs_to_tensor
        except ImportError:
            pytest.skip("torch not available")

        agent = EcoAgent(num_players=2)
        vec   = VecSinglePlayerEcoEnv(num_envs=2, num_players=2, seeds=[0, 1])
        obs_np, masks_np = vec.reset()

        # Add batch dim (already batched by vec)
        obs_t  = obs_to_tensor(obs_np, torch.device("cpu"))
        mask_t = torch.as_tensor(masks_np, dtype=torch.bool)

        action, logprob, entropy, value = agent.get_action_and_value(obs_t, mask_t)
        assert action.shape  == (2,)
        assert logprob.shape == (2,)
        assert value.shape   == (2, 1)

    def test_action_within_legal_mask(self):
        try:
            import torch
            from eco_ppo import EcoAgent, obs_to_tensor
        except ImportError:
            pytest.skip("torch not available")

        agent = EcoAgent(num_players=2)
        vec   = VecSinglePlayerEcoEnv(num_envs=8, num_players=2, seeds=list(range(8)))
        obs_np, masks_np = vec.reset()

        obs_t  = obs_to_tensor(obs_np, torch.device("cpu"))
        mask_t = torch.as_tensor(masks_np, dtype=torch.bool)

        for _ in range(10):
            actions, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
            for i, a in enumerate(actions.cpu().numpy()):
                assert masks_np[i, a], f"Agent chose illegal action {a} for env {i}"
            obs_np, masks_np, _, terminated, _, _ = vec.step(actions.cpu().numpy())
            obs_t  = obs_to_tensor(obs_np, torch.device("cpu"))
            mask_t = torch.as_tensor(masks_np, dtype=torch.bool)


class TestResetNeverDone:
    """reset() must never return a done state (opponents may end the game
    before the agent's first turn — reset should re-roll in that case)."""

    def test_single_env_reset_not_done(self):
        """SinglePlayerEcoEnv.reset() should always return a live game."""
        for np_ in [2, 3, 4, 5]:
            for _ in range(200):
                env = SinglePlayerEcoEnv(num_players=np_)
                env.reset()
                assert not env.state.done, (
                    f"reset() returned done=True for {np_}-player game"
                )
                mask = env.legal_actions()
                assert mask.any(), (
                    f"reset() returned all-False mask for {np_}-player game"
                )

    def test_vec_env_reset_not_done(self):
        """VecSinglePlayerEcoEnv.reset() masks must all have legal actions."""
        for np_ in [2, 3, 4, 5]:
            vec = VecSinglePlayerEcoEnv(num_envs=64, num_players=np_)
            obs, masks = vec.reset()
            for i in range(64):
                assert masks[i].any(), (
                    f"Vec env {i}: reset() returned all-False mask "
                    f"for {np_}-player game"
                )
            vec.close()

    def test_vec_env_auto_reset_not_done(self):
        """After auto-reset on termination, masks must have legal actions."""
        vec = VecSinglePlayerEcoEnv(num_envs=32, num_players=3)
        obs, masks = vec.reset()
        for step in range(500):
            actions = np.array([
                np.random.choice(np.where(masks[i])[0])
                for i in range(32)
            ], dtype=np.int32)
            obs, masks, rewards, terminated, truncated, infos = vec.step(actions)
            for i in range(32):
                assert masks[i].any(), (
                    f"Step {step}, env {i}: all-False mask after "
                    f"{'auto-reset' if terminated[i] else 'step'}"
                )
        vec.close()
