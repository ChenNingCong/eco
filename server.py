"""
server.py — Flask server to play Hearts against AI opponents.

Usage:
    python server.py                        # random opponents, no trained agent
    python server.py --model model/latest.pkt
    python server.py --model model/latest.pkt --port 5000

The server watches the model directory and always loads the newest .pkt file
so you don't need to restart between training runs.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):    return o.tolist()
        return super().default(o)

sys.path.insert(0, os.path.dirname(__file__))

from hearts_env import (
    HeartsEnv, NUM_PLAYERS, NUM_CARDS, MAX_SCORE,
    card_suit, card_rank, card_points,
    CLUBS, DIAMONDS, HEARTS, SPADES,
    TWO_OF_CLUBS, QUEEN_OF_SPADES,
)

# ── Card helpers ──────────────────────────────────────────────────────────────

RANK_NAMES  = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
SUIT_NAMES  = {CLUBS: 'clubs', DIAMONDS: 'diamonds', HEARTS: 'hearts', SPADES: 'spades'}

def card_to_dict(card_id: int) -> dict:
    return {
        "id":     int(card_id),
        "rank":   RANK_NAMES[card_rank(card_id)],
        "suit":   SUIT_NAMES[card_suit(card_id)],
        "points": int(card_points(card_id)),
    }

# ── Agent loading ─────────────────────────────────────────────────────────────

_agent_cache = {"path": None, "agent": None, "mtime": 0}

def _newest_model(model_dir: str):
    """Return path to the newest .pkt file in model_dir, or None."""
    files = glob.glob(os.path.join(model_dir, "*.pkt"))
    return max(files, key=os.path.getmtime) if files else None

def get_trained_agent(model_dir: str):
    """Load (and cache) the newest trained agent. Returns None if unavailable."""
    try:
        import torch
        from obs_encoder import PyTreeObs
        from ppo import Agent, obs_to_tensor
    except ImportError:
        return None

    path = _newest_model(model_dir)
    if path is None:
        return None

    mtime = os.path.getmtime(path)
    if _agent_cache["path"] == path and _agent_cache["mtime"] == mtime:
        return _agent_cache["agent"]

    try:
        device = torch.device("cpu")
        agent = Agent().to(device)
        agent.load_state_dict(torch.load(path, map_location=device))
        agent.eval()
        _agent_cache.update({"path": path, "agent": agent, "mtime": mtime})
        print(f"[server] Loaded model: {path}")
        return agent
    except Exception as e:
        print(f"[server] Failed to load model {path}: {e}")
        return None

# ── Heuristic agents (copied from ppo.py to avoid circular import) ────────────

class RandomAgent:
    name = "random"
    def act(self, mask: np.ndarray) -> int:
        legal = np.where(mask)[0]
        return int(np.random.choice(legal))

class AvoidPointsAgent:
    name = "avoid_points"
    def act(self, mask: np.ndarray) -> int:
        legal = sorted(np.where(mask)[0], key=lambda c: (card_points(c), card_rank(c)))
        return int(legal[0])

class BleedHeartsAgent:
    name = "bleed_hearts"
    def act(self, mask: np.ndarray) -> int:
        legal = np.where(mask)[0]
        if QUEEN_OF_SPADES in legal:
            return int(QUEEN_OF_SPADES)
        legal = sorted(legal, key=lambda c: (card_points(c), card_rank(c)), reverse=True)
        return int(legal[0])

class SafeAgent:
    name = "safe"
    def act(self, mask: np.ndarray) -> int:
        legal = list(np.where(mask)[0])
        safe  = [c for c in legal if card_points(c) == 0]
        if safe:
            return int(max(safe, key=card_rank))
        return int(min(legal, key=lambda c: (card_points(c), card_rank(c))))

class TrainedAgent:
    name = "trained"
    def __init__(self, model_dir):
        self.model_dir = model_dir
        # SinglePlayerEnv wrapper used purely for its _encode_for helper
        self._wrapper = None

    def _get_wrapper(self, env: HeartsEnv):
        """Lazily create a SinglePlayerEnv wrapper around the raw env."""
        from obs_encoder import SinglePlayerEnv
        if self._wrapper is None:
            self._wrapper = SinglePlayerEnv.__new__(SinglePlayerEnv)
            self._wrapper.env = env
            self._wrapper._seat = env.state.current_player
        else:
            self._wrapper.env = env
            self._wrapper._seat = env.state.current_player
        return self._wrapper

    def act(self, mask: np.ndarray, env: HeartsEnv) -> int:
        try:
            import torch
            from obs_encoder import PyTreeObs
            from ppo import Agent, obs_to_tensor
        except ImportError:
            return int(np.random.choice(np.where(mask)[0]))

        agent = get_trained_agent(self.model_dir)
        if agent is None:
            return int(np.random.choice(np.where(mask)[0]))

        wrapper = self._get_wrapper(env)
        obs = wrapper._encode_for(env.state.current_player)
        obs_t  = obs_to_tensor(PyTreeObs(*[np.expand_dims(f, 0) for f in obs]),
                               torch.device("cpu"))
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
        return int(action.item())

# ── Game session ──────────────────────────────────────────────────────────────

class GameSession:
    def __init__(self, human_seat: int, opponent_types: dict, model_dir: str):
        """
        opponent_types: dict mapping seat -> type string, e.g. {1:'random', 2:'safe', 3:'trained'}
        """
        self.human_seat    = human_seat
        self.opponent_types = opponent_types  # seat -> type string
        self.model_dir     = model_dir
        self.env           = HeartsEnv(seed=int(time.time()))
        self.state         = self.env.reset()
        self.pending_events = []  # events queued for frontend to animate

        # Build opponent agents for the 3 non-human seats
        self.opponents = {}
        for seat in range(NUM_PLAYERS):
            if seat != human_seat:
                t = opponent_types.get(seat, 'random')
                self.opponents[seat] = self._make_opponent(t, model_dir)

        # Auto-play until it's the human's turn
        self._advance()

    def _make_opponent(self, opp_type: str, model_dir: str):
        if opp_type == "avoid_points":
            return AvoidPointsAgent()
        elif opp_type == "bleed_hearts":
            return BleedHeartsAgent()
        elif opp_type == "safe":
            return SafeAgent()
        elif opp_type == "trained":
            return TrainedAgent(model_dir)
        else:
            return RandomAgent()

    def _advance(self):
        """Play AI turns, collecting events, until it's the human's turn or game over."""
        while not self.state.done and self.state.current_player != self.human_seat:
            p    = self.state.current_player
            mask = self.env.legal_actions()
            opp  = self.opponents[p]
            trick_count_before = self.state.current_trick_count
            if isinstance(opp, TrainedAgent):
                action = opp.act(mask, self.env)
            else:
                action = opp.act(mask)
            prev_state = self.state
            self.state, rewards, done, _ = self.env.step(action)
            self.pending_events.append({
                "type":   "play",
                "player": p,
                "card":   card_to_dict(action),
            })
            # Did a trick just complete?
            if trick_count_before == NUM_PLAYERS - 1 or done:
                if self.state.history:
                    rec = self.state.history[-1]
                    self.pending_events.append({
                        "type":   "trick_complete",
                        "winner": int(rec.winner),
                        "points": int(sum(card_points(c) for c in rec.cards if c >= 0)),
                    })
            if done:
                self._append_result()
                break

    def _append_result(self):
        scores = self.state.scores.tolist()
        winner = int(np.argmin(scores))
        self.pending_events.append({
            "type":    "game_over",
            "scores":  scores,
            "winner":  winner,
        })

    def human_play(self, card_id: int) -> dict:
        """Process a human card play. Returns updated game state + animation events."""
        mask = self.env.legal_actions()
        if not mask[card_id]:
            return {"error": "Illegal card"}

        self.pending_events = []
        trick_count_before = self.state.current_trick_count
        self.state, rewards, done, _ = self.env.step(card_id)
        self.pending_events.append({
            "type":   "play",
            "player": self.human_seat,
            "card":   card_to_dict(card_id),
        })
        if trick_count_before == NUM_PLAYERS - 1 or done:
            if self.state.history:
                rec = self.state.history[-1]
                self.pending_events.append({
                    "type":   "trick_complete",
                    "winner": int(rec.winner),
                    "points": int(sum(card_points(c) for c in rec.cards if c >= 0)),
                })
        if done:
            self._append_result()
        else:
            self._advance()

        return self.to_dict()

    def to_dict(self) -> dict:
        s = self.state
        legal = self.env.legal_actions().tolist() if not s.done else [False] * NUM_CARDS

        # Current trick
        trick = []
        for i in range(s.current_trick_count):
            trick.append({
                "player": int(s.current_trick_players[i]),
                "card":   card_to_dict(int(s.current_trick_cards[i])),
            })

        # Last completed trick from history
        last_trick = None
        if s.history:
            rec = s.history[-1]
            last_trick = {
                "winner": int(rec.winner),
                "cards": [
                    {"player": int(rec.players[i]), "card": card_to_dict(int(rec.cards[i]))}
                    for i in range(NUM_PLAYERS)
                ]
            }

        # Per-seat opponent labels
        seat_labels = {}
        for seat in range(NUM_PLAYERS):
            if seat == self.human_seat:
                seat_labels[seat] = "human"
            else:
                seat_labels[seat] = self.opponent_types.get(seat, "random")

        events = list(self.pending_events)
        self.pending_events = []

        return {
            "human_seat":     self.human_seat,
            "current_player": int(s.current_player),
            "hand":           [card_to_dict(c) for c in np.where(s.hands[self.human_seat])[0]],
            "scores":         s.scores.tolist(),
            "round_num":      int(s.round_num),
            "hearts_broken":  bool(s.hearts_broken),
            "legal_ids":      [int(c) for c in np.where(legal)[0]],
            "current_trick":  trick,
            "last_trick":     last_trick,
            "done":           bool(s.done),
            "events":         events,
            "seat_labels":    seat_labels,
        }


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
sessions: dict[str, GameSession] = {}
MODEL_DIR = "model"

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/new_game", methods=["POST"])
def new_game():
    data       = request.json or {}
    # opponents: dict of seat(str) -> type, e.g. {"1":"random","2":"safe","3":"trained"}
    # Falls back to a single "opponent" key for backwards compat
    raw = data.get("opponents", {})
    default = data.get("opponent", "random")
    opponent_types = {
        seat: raw.get(str(seat), default)
        for seat in [1, 2, 3]
    }
    session_id = str(int(time.time() * 1000))
    sess = GameSession(
        human_seat=0,
        opponent_types=opponent_types,
        model_dir=MODEL_DIR,
    )
    sessions[session_id] = sess
    return jsonify({"session_id": session_id, **sess.to_dict()})

@app.route("/api/play", methods=["POST"])
def play():
    data       = request.json or {}
    session_id = data.get("session_id")
    card_id    = data.get("card_id")
    sess = sessions.get(session_id)
    if sess is None:
        return jsonify({"error": "Session not found"}), 404
    result = sess.human_play(int(card_id))
    return jsonify({"session_id": session_id, **result})

@app.route("/api/model_status")
def model_status():
    path = _newest_model(MODEL_DIR)
    if path is None:
        return jsonify({"available": False})
    mtime = os.path.getmtime(path)
    return jsonify({
        "available": True,
        "path": os.path.basename(path),
        "mtime": int(mtime),
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="model", help="Directory with .pkt model files")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    MODEL_DIR = args.model_dir
    os.makedirs("static", exist_ok=True)
    print(f"[server] Starting on http://{args.host}:{args.port}")
    print(f"[server] Model dir: {os.path.abspath(MODEL_DIR)}")
    app.run(host=args.host, port=args.port, debug=False)