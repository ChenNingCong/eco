"""
SinglePlayerEnv — wraps a multi-player game engine into a single-agent env.

The agent plays one seat; all other seats are driven by an opponent function.
From the agent's perspective, each step() is one of its own action decisions.
Opponents act internally between agent turns.

This is NOT customizable — it's a fixed wrapper over any BaseGameEngine.
"""

from typing import Generic
import numpy as np
from .game import BaseGameEngine, Obs
from .player import SlicedPlayer


class SinglePlayerEnv(Generic[Obs]):
    """
    Single-agent wrapper for a multi-player BaseGameEngine.

    Parameters
    ----------
    engine   : the game engine (with its own RNG)
    opponent : SlicedPlayer for opponent turns
    rng      : np.random.Generator for env-level randomness (seat selection)
    """

    def __init__(self, engine: BaseGameEngine[Obs],
                 opponent: SlicedPlayer,
                 rng: np.random.Generator):
        self.engine: BaseGameEngine[Obs] = engine
        self.opponent: SlicedPlayer = opponent
        self._rng: np.random.Generator = rng

    # ── Public API (gym 0.26 style) ──────────────────────────────────────

    def reset(self, *, rng: np.random.Generator | None = None, _retries: int = 0) -> tuple[Obs, dict]:
        if _retries > 100:
            raise RuntimeError(
                "SinglePlayerEnv.reset(): game ended before agent's turn "
                "100 times in a row — likely a bug in the game engine")
        if rng is not None:
            engine_rng, env_rng = rng.spawn(2)
            self.engine.reset(rng=engine_rng)
            self._rng = env_rng
        else:
            self.engine.reset()
        # Randomly assign agent to any seat for training symmetry
        self._seat = int(self._rng.integers(self.engine.num_players))
        self._accumulated_reward = 0.0

        # If agent is not first to move, run opponents until agent's turn
        while self.engine.current_player != self._seat and not self.engine.done:
            opp_obs = self.engine.encode(self.engine.current_player)
            opp_mask = self.engine.legal_actions()
            opp_action = self.opponent.action(opp_obs, opp_mask)
            rewards = self.engine.step(opp_action)
            self._accumulate(rewards)

        # If opponents ended the game before agent's turn, re-reset
        if self.engine.done:
            return self.reset(_retries=_retries + 1)

        return self.engine.encode(self._seat), {}

    def step(self, action: int) -> tuple[Obs, float, bool, bool, dict]:
        assert not self.engine.done

        rewards = self.engine.step(action)
        self._accumulate(rewards)

        if self.engine.done:
            return self._terminal_return()

        # Advance opponents until agent's turn
        while self.engine.current_player != self._seat:
            opp_obs = self.engine.encode(self.engine.current_player)
            opp_mask = self.engine.legal_actions()
            opp_action = self.opponent.action(opp_obs, opp_mask)
            rewards = self.engine.step(opp_action)
            self._accumulate(rewards)
            if self.engine.done:
                return self._terminal_return()

        reward = self._accumulated_reward
        self._accumulated_reward = 0.0
        return self.engine.encode(self._seat), reward, False, False, {}

    def step_gen(self, action: int):
        """
        Generator version of step() for batched opponent evaluation.

        Yields (obs, mask) for each opponent decision. The caller sends
        back the action via gen.send(action). When the generator returns,
        the return value is the usual (obs, reward, terminated, truncated, info).
        """
        assert not self.engine.done

        rewards = self.engine.step(action)
        self._accumulate(rewards)

        if self.engine.done:
            return self._terminal_return()

        while self.engine.current_player != self._seat:
            opp_obs = self.engine.encode(self.engine.current_player)
            opp_mask = self.engine.legal_actions()
            opp_action = yield (opp_obs, opp_mask)

            rewards = self.engine.step(int(opp_action))
            self._accumulate(rewards)
            if self.engine.done:
                return self._terminal_return()

        reward = self._accumulated_reward
        self._accumulated_reward = 0.0
        return self.engine.encode(self._seat), reward, False, False, {}

    def legal_actions(self) -> np.ndarray:
        return self.engine.legal_actions()

    # ── Helpers ───────────────────────────────────────────────────────

    def _accumulate(self, rewards: tuple):
        """Accumulate reward for the agent's seat."""
        self._accumulated_reward += rewards[self._seat]

    def _terminal_return(self):
        reward = self._accumulated_reward
        self._accumulated_reward = 0.0
        obs = self.engine.encode(self._seat)
        info: dict = {"agent_seat": self._seat}
        try:
            info["final_scores"] = self.engine.compute_scores()
        except NotImplementedError:
            pass
        try:
            info["game_metrics"] = self.engine.game_metrics(self._seat)
        except (NotImplementedError, AttributeError):
            pass
        return obs, reward, True, False, info
