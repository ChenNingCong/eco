"""
server.py — Flask server to play R-öko against AI opponents.

Usage:
    python server.py                        # random opponent
    python server.py --model-dir model      # with trained agent
    python server.py --port 5000
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

# ── Agent loading (Eco) ─────────────────────────────────────────────────────

_eco_agent_cache = {"path": None, "agent": None, "mtime": 0, "num_players": None}

def get_eco_agent(model_dir: str, num_players: int = 2, model_file: str = "latest"):
    """Load (and cache) a trained Eco agent. Returns None if unavailable."""
    try:
        import torch
        from eco_ppo import EcoAgent, obs_to_tensor
        from eco_obs_encoder import EcoPyTreeObs
    except ImportError:
        return None
    if model_file == "latest":
        files = glob.glob(os.path.join(model_dir, "eco_*.pkt"))
        if not files:
            return None
        path = max(files, key=os.path.getmtime)
    else:
        path = os.path.join(model_dir, model_file)
        if not os.path.exists(path):
            return None
    mtime = os.path.getmtime(path)
    if (_eco_agent_cache["path"] == path and _eco_agent_cache["mtime"] == mtime
            and _eco_agent_cache["num_players"] == num_players):
        return _eco_agent_cache["agent"]
    device = torch.device("cpu")
    for sc in [False, True]:
        try:
            agent  = EcoAgent(num_players=num_players, score_shortcut=sc).to(device)
            agent.load_state_dict(torch.load(path, map_location=device))
            agent.eval()
            _eco_agent_cache.update({"path": path, "agent": agent, "mtime": mtime,
                                     "num_players": num_players})
            print(f"[server] Loaded eco model: {path} (score_shortcut={sc})")
            return agent
        except Exception:
            continue
    print(f"[server] Failed to load eco model {path}")
    return None


COLOR_NAMES = ["Glass", "Paper", "Plastic", "Tin"]
COLOR_KEYS  = ["glass", "paper", "plastic", "tin"]
TYPE_NAMES  = ["single", "double"]
TYPE_VALUES = [1, 2]


# ── R-öko game session ──────────────────────────────────────────────────────

class EcoGameSession:
    """Manages a multi-player R-öko game between a human and AI opponents."""

    def __init__(self, num_players: int, opponent_type: str, model_dir: str,
                 model_file: str = "latest", human_seat: int = 0):
        from eco_env import EcoEnv
        self.num_players    = num_players
        self.model_dir      = model_dir
        self.model_file     = model_file
        self.opponent_type  = opponent_type
        self.env            = EcoEnv(num_players=num_players, seed=int(time.time()))
        self.state          = self.env.reset()
        self.human_seat     = human_seat
        self.pending_events = []
        self._advance()

    def _opponent_act(self) -> int:
        mask = self.env.legal_actions()
        if self.opponent_type == "trained":
            agent = get_eco_agent(self.model_dir, self.num_players, self.model_file)
            if agent is not None:
                try:
                    import torch
                    from eco_obs_encoder import SinglePlayerEcoEnv, EcoPyTreeObs
                    from eco_ppo import obs_to_tensor
                    wrapper = SinglePlayerEcoEnv.__new__(SinglePlayerEcoEnv)
                    wrapper.env            = self.env
                    wrapper._num_players   = self.num_players
                    wrapper._seat          = self.state.current_player
                    wrapper._relative_seat = True
                    obs   = wrapper._encode_for(self.state.current_player)
                    obs_t = obs_to_tensor(
                        EcoPyTreeObs(*[np.expand_dims(f, 0) for f in obs]),
                        torch.device("cpu"))
                    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
                    return int(action.item())
                except Exception:
                    pass
        # fallback: random
        return int(np.random.choice(np.where(mask)[0]))

    def _advance(self):
        """Drive opponent turns until it's the human's turn (or game over).
        Collects intermediate state snapshots so the frontend can replay them."""
        s = self.state
        while not s.done and s.current_player != self.human_seat:
            actor = int(s.current_player)
            pre_snap = self._snapshot()
            action = self._opponent_act()
            self.state, _, done, _ = self.env.step(action)
            self.pending_events.append({
                "type": "opponent_action",
                "action": int(action),
                "player": actor,
                "pre_snapshot": pre_snap,
                "snapshot": self._snapshot(),
            })
            if done:
                self.pending_events.append({"type": "game_over",
                                            "scores": self.env.compute_scores(self.state).tolist()})
                break
            s = self.state

    def _snapshot(self) -> dict:
        """Lightweight state snapshot for opponent action replay."""
        from eco_env import NUM_COLORS
        s = self.state
        np_ = self.num_players
        hands = []
        for p in range(np_):
            row = []
            for c in range(NUM_COLORS):
                row.append({"color": COLOR_KEYS[c],
                            "single": int(s.hands[p, c, 0]),
                            "double": int(s.hands[p, c, 1])})
            hands.append(row)
        factories = []
        for c in range(NUM_COLORS):
            rec_val = int(s.recycling_side[c, 0] * 1 + s.recycling_side[c, 1] * 2)
            factories.append({
                "color": COLOR_KEYS[c],
                "recycling_single": int(s.recycling_side[c, 0]),
                "recycling_double": int(s.recycling_side[c, 1]),
                "recycling_value": rec_val,
                "waste": [
                    {"color": COLOR_KEYS[cc],
                     "single": int(s.waste_side[c, cc, 0]),
                     "double": int(s.waste_side[c, cc, 1])}
                    for cc in range(NUM_COLORS)
                    if s.waste_side[c, cc].sum() > 0
                ],
                "stack_remaining": len(s.factory_stacks[c]),
                "stack_top": s.factory_stacks[c][0] if s.factory_stacks[c] else None,
            })
        collected = []
        for p in range(np_):
            row = []
            for c in range(NUM_COLORS):
                cards = s.collected[p][c]
                row.append({"color": COLOR_KEYS[c], "cards": list(cards),
                            "total": sum(cards),
                            "counts": True if len(cards) > 1 else False})
            collected.append(row)
        penalty = []
        for p in range(np_):
            pile = []
            for c in range(NUM_COLORS):
                if s.penalty_pile[p, c, 0] > 0:
                    pile.append({"color": COLOR_KEYS[c], "type": "single",
                                 "count": int(s.penalty_pile[p, c, 0])})
                if s.penalty_pile[p, c, 1] > 0:
                    pile.append({"color": COLOR_KEYS[c], "type": "double",
                                 "count": int(s.penalty_pile[p, c, 1])})
            penalty.append(pile)
        return {
            "current_player": int(s.current_player),
            "phase": int(s.phase),
            "done": bool(s.done),
            "draw_pile_size": len(s.draw_pile),
            "hands": hands,
            "factories": factories,
            "collected": collected,
            "penalty": penalty,
            "scores": self.env.compute_scores(s).tolist(),
        }

    def human_action(self, action_id: int) -> dict:
        mask = self.env.legal_actions()
        if not mask[action_id]:
            return {"error": "Illegal action"}
        self.pending_events = []
        self.state, _, done, _ = self.env.step(action_id)
        self.pending_events.append({"type": "human_action", "action": int(action_id)})
        if done:
            self.pending_events.append({"type": "game_over",
                                        "scores": self.env.compute_scores(self.state).tolist()})
        else:
            self._advance()
        return self.to_dict()

    def to_dict(self) -> dict:
        from eco_env import (NUM_COLORS, NUM_TYPES, NUM_ACTIONS, PHASE_PLAY,
                             PHASE_DISCARD, encode_play, encode_discard, _STACK)
        s    = self.state
        np_  = self.num_players
        mask = self.env.legal_actions().tolist() if not s.done else [False] * NUM_ACTIONS

        # Hands for all players
        hands = []
        for p in range(np_):
            row = []
            for c in range(NUM_COLORS):
                row.append({
                    "color": COLOR_KEYS[c],
                    "single": int(s.hands[p, c, 0]),
                    "double": int(s.hands[p, c, 1]),
                })
            hands.append(row)

        # Factories
        factories = []
        for c in range(NUM_COLORS):
            rec_val = int(s.recycling_side[c, 0] * 1 + s.recycling_side[c, 1] * 2)
            factories.append({
                "color":           COLOR_KEYS[c],
                "recycling_single": int(s.recycling_side[c, 0]),
                "recycling_double": int(s.recycling_side[c, 1]),
                "recycling_value":  rec_val,
                "waste":           [
                    {"color": COLOR_KEYS[cc],
                     "single": int(s.waste_side[c, cc, 0]),
                     "double": int(s.waste_side[c, cc, 1])}
                    for cc in range(NUM_COLORS)
                    if s.waste_side[c, cc].sum() > 0
                ],
                "stack_remaining": len(s.factory_stacks[c]),
                "stack_top":       s.factory_stacks[c][0] if s.factory_stacks[c] else None,
            })

        # Collected factory cards per player per color
        collected = []
        for p in range(np_):
            row = []
            for c in range(NUM_COLORS):
                cards = s.collected[p][c]
                row.append({
                    "color":  COLOR_KEYS[c],
                    "cards":  list(cards),
                    "total":  sum(cards),
                    "counts": True if len(cards) > 1 else False,
                })
            collected.append(row)

        # Penalty piles
        penalty = []
        for p in range(np_):
            pile = []
            for c in range(NUM_COLORS):
                if s.penalty_pile[p, c, 0] > 0:
                    pile.append({"color": COLOR_KEYS[c], "type": "single",
                                 "count": int(s.penalty_pile[p, c, 0])})
                if s.penalty_pile[p, c, 1] > 0:
                    pile.append({"color": COLOR_KEYS[c], "type": "double",
                                 "count": int(s.penalty_pile[p, c, 1])})
            penalty.append(pile)

        # Legal play actions for human
        legal_plays = []
        legal_discards = []
        if not s.done and s.current_player == self.human_seat:
            from eco_env import NUM_PLAY_ACTIONS, decode_play, decode_discard
            if s.phase == PHASE_PLAY:
                for a, legal in enumerate(mask[:NUM_PLAY_ACTIONS]):
                    if legal:
                        c, ns, nd = decode_play(a)
                        legal_plays.append({
                            "action_id": a,
                            "color": COLOR_KEYS[c],
                            "single": ns,
                            "double": nd,
                        })
            else:
                for a, legal in enumerate(mask[NUM_PLAY_ACTIONS:], start=NUM_PLAY_ACTIONS):
                    if legal:
                        c, t = decode_discard(a)
                        legal_discards.append({
                            "action_id": a,
                            "color": COLOR_KEYS[c],
                            "type": TYPE_NAMES[t],
                        })

        events = list(self.pending_events)
        self.pending_events = []

        return {
            "num_players":   np_,
            "human_seat":    self.human_seat,
            "current_player": int(s.current_player),
            "phase":          int(s.phase),
            "done":           bool(s.done),
            "draw_pile_size": len(s.draw_pile),
            "hands":          hands,
            "factories":      factories,
            "collected":      collected,
            "penalty":        penalty,
            "legal_plays":    legal_plays,
            "legal_discards": legal_discards,
            "scores":         self.env.compute_scores(s).tolist(),
            "events":         events,
            "opponent_type":  self.opponent_type,
        }


# ── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
eco_sessions: dict[str, EcoGameSession] = {}
MODEL_DIR = "model"

@app.route("/")
def index():
    return send_from_directory("static", "eco_index.html")

@app.route("/api/eco/new_game", methods=["POST"])
def eco_new_game():
    data         = request.json or {}
    num_players  = int(data.get("num_players", 2))
    opponent     = data.get("opponent", "random")
    model_file   = data.get("model_file", "latest")
    human_seat   = int(data.get("human_seat", 0))
    human_seat   = max(0, min(human_seat, num_players - 1))
    session_id   = str(int(time.time() * 1000))
    sess         = EcoGameSession(num_players=num_players, opponent_type=opponent,
                                  model_dir=MODEL_DIR, model_file=model_file,
                                  human_seat=human_seat)
    eco_sessions[session_id] = sess
    return jsonify({"session_id": session_id, **sess.to_dict()})

@app.route("/api/eco/play", methods=["POST"])
def eco_play():
    data       = request.json or {}
    session_id = data.get("session_id")
    action_id  = data.get("action_id")
    sess = eco_sessions.get(session_id)
    if sess is None:
        return jsonify({"error": "Session not found"}), 404
    result = sess.human_action(int(action_id))
    return jsonify({"session_id": session_id, **result})

@app.route("/api/eco/model_status")
def eco_model_status():
    files = glob.glob(os.path.join(MODEL_DIR, "eco_*.pkt"))
    if not files:
        return jsonify({"available": False, "models": []})
    # List all models sorted by step number (newest first)
    models = []
    for f in files:
        base = os.path.basename(f)
        name = base.replace("eco_", "").replace(".pkt", "")
        models.append({"file": base, "name": name})
    # Sort: "latest" first, then by step descending
    def sort_key(m):
        if m["name"] == "latest":
            return float('inf')
        try:
            return int(m["name"])
        except ValueError:
            return -1
    models.sort(key=sort_key, reverse=True)
    path  = max(files, key=os.path.getmtime)
    return jsonify({"available": True, "path": os.path.basename(path),
                    "models": models})

# ── Spectator session (all-AI) ─────────────────────────────────────────────

class SpectatorSession:
    """All-AI game session for watching different models play each other."""

    def __init__(self, num_players: int, model_files: list, model_dir: str):
        from eco_env import EcoEnv
        self.num_players = num_players
        self.model_dir = model_dir
        self.model_files = model_files  # one per seat
        self.env = EcoEnv(num_players=num_players, seed=int(time.time()))
        self.state = self.env.reset()

    def _act(self, seat: int) -> int:
        mask = self.env.legal_actions()
        model_file = self.model_files[seat]
        if model_file and model_file != "random":
            agent = get_eco_agent(self.model_dir, self.num_players, model_file)
            if agent is not None:
                try:
                    import torch
                    from eco_obs_encoder import SinglePlayerEcoEnv, EcoPyTreeObs
                    from eco_ppo import obs_to_tensor
                    wrapper = SinglePlayerEcoEnv.__new__(SinglePlayerEcoEnv)
                    wrapper.env = self.env
                    wrapper._num_players = self.num_players
                    wrapper._seat = seat
                    obs = wrapper._encode_for(seat)
                    obs_t = obs_to_tensor(
                        EcoPyTreeObs(*[np.expand_dims(f, 0) for f in obs]),
                        torch.device("cpu"))
                    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
                    return int(action.item())
                except Exception:
                    pass
        return int(np.random.choice(np.where(mask)[0]))

    def _snapshot(self) -> dict:
        from eco_env import NUM_COLORS
        s = self.state
        np_ = self.num_players
        hands = []
        for p in range(np_):
            row = []
            for c in range(NUM_COLORS):
                row.append({"color": COLOR_KEYS[c],
                            "single": int(s.hands[p, c, 0]),
                            "double": int(s.hands[p, c, 1])})
            hands.append(row)
        factories = []
        for c in range(NUM_COLORS):
            rec_val = int(s.recycling_side[c, 0] * 1 + s.recycling_side[c, 1] * 2)
            factories.append({
                "color": COLOR_KEYS[c],
                "recycling_single": int(s.recycling_side[c, 0]),
                "recycling_double": int(s.recycling_side[c, 1]),
                "recycling_value": rec_val,
                "waste": [
                    {"color": COLOR_KEYS[cc],
                     "single": int(s.waste_side[c, cc, 0]),
                     "double": int(s.waste_side[c, cc, 1])}
                    for cc in range(NUM_COLORS)
                    if s.waste_side[c, cc].sum() > 0
                ],
                "stack_remaining": len(s.factory_stacks[c]),
                "stack_top": s.factory_stacks[c][0] if s.factory_stacks[c] else None,
            })
        collected = []
        for p in range(np_):
            row = []
            for c in range(NUM_COLORS):
                cards = s.collected[p][c]
                row.append({"color": COLOR_KEYS[c], "cards": list(cards),
                            "total": sum(cards),
                            "counts": True if len(cards) > 1 else False})
            collected.append(row)
        penalty = []
        for p in range(np_):
            pile = []
            for c in range(NUM_COLORS):
                if s.penalty_pile[p, c, 0] > 0:
                    pile.append({"color": COLOR_KEYS[c], "type": "single",
                                 "count": int(s.penalty_pile[p, c, 0])})
                if s.penalty_pile[p, c, 1] > 0:
                    pile.append({"color": COLOR_KEYS[c], "type": "double",
                                 "count": int(s.penalty_pile[p, c, 1])})
            penalty.append(pile)
        return {
            "current_player": int(s.current_player),
            "phase": int(s.phase),
            "done": bool(s.done),
            "draw_pile_size": len(s.draw_pile),
            "hands": hands,
            "factories": factories,
            "collected": collected,
            "penalty": penalty,
            "scores": self.env.compute_scores(s).tolist(),
        }

    def step(self) -> dict:
        """Execute one player's action. Returns action info + snapshot."""
        if self.state.done:
            return {"done": True, "snapshot": self._snapshot(),
                    "scores": self.env.compute_scores(self.state).tolist()}
        seat = int(self.state.current_player)
        pre_snap = self._snapshot()
        action = self._act(seat)
        self.state, _, done, _ = self.env.step(action)
        post_snap = self._snapshot()
        result = {
            "done": bool(self.state.done),
            "action": int(action),
            "player": seat,
            "pre_snapshot": pre_snap,
            "snapshot": post_snap,
        }
        if self.state.done:
            result["scores"] = self.env.compute_scores(self.state).tolist()
        return result


spectator_sessions: dict[str, SpectatorSession] = {}


@app.route("/api/eco/spectate/new_game", methods=["POST"])
def eco_spectate_new():
    data = request.json or {}
    num_players = int(data.get("num_players", 3))
    model_files = data.get("model_files", ["latest"] * num_players)
    # Pad or trim to num_players
    while len(model_files) < num_players:
        model_files.append("random")
    model_files = model_files[:num_players]
    session_id = "spec_" + str(int(time.time() * 1000))
    sess = SpectatorSession(num_players=num_players, model_files=model_files,
                            model_dir=MODEL_DIR)
    spectator_sessions[session_id] = sess
    return jsonify({"session_id": session_id, "num_players": num_players,
                    "model_files": model_files, "snapshot": sess._snapshot()})


@app.route("/api/eco/spectate/step", methods=["POST"])
def eco_spectate_step():
    data = request.json or {}
    session_id = data.get("session_id")
    sess = spectator_sessions.get(session_id)
    if sess is None:
        return jsonify({"error": "Session not found"}), 404
    result = sess.step()
    return jsonify({"session_id": session_id, **result})


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
    app.run(host=args.host, port=args.port, debug=True)
