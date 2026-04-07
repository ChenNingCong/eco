"""
server.py — Flask server to play Ticket to Ride against AI.

Usage:
    python -m game.ttr.server                        # random opponent
    python -m game.ttr.server --model-dir model/ttr  # with trained agent
    python -m game.ttr.server --port 5000
"""

import argparse
import os
import time

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from game.ttr.engine import (
    TTREngine, TTRObs,
    GAME_FIRST_ROUND, GAME_PLAYING, GAME_LAST_ROUND, GAME_OVER,
    TURN_INIT, TURN_SELECTING_DEST, TURN_DRAWING_CARDS, TURN_FINISHED,
    ACT_DRAW_RANDOM, ACT_DRAW_VISIBLE_BASE, ACT_DRAW_DEST,
    ACT_SELECT_DEST_BASE, ACT_FINISH_SELECT, ACT_CLAIM_ROUTE_BASE,
    NUM_ACTIONS,
)
from game.ttr.map_data import (
    CITIES, NUM_COLORS, COLOR_NAMES, WILD,
    NUM_ROUTES, ROUTE_CITY1, ROUTE_CITY2, ROUTE_COLOR, ROUTE_LENGTH,
    ROUTE_ADJACENT, ROUTE_POINTS_LIST,
    NUM_DESTINATIONS, DEST_CITY1, DEST_CITY2, DEST_POINTS,
    STARTING_TRAINS, CITY_COORDS,
)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=_STATIC_DIR)


# ── Game session ─────────────────────────────────────────────────────────────

class GameSession:
    def __init__(self, ai_fn=None):
        self.ai_fn = ai_fn  # callable(engine, player_id, lstm_state) -> (action, new_lstm_state), or None
        self.engine = None
        self.rng = None
        self.lstm_state = None     # persistent LSTM state for AI across a game
        self.last_ai_actions = []  # actions AI took on its last sequence of turns
        self.action_log = []       # full game action log

    def new_game(self, seed=None):
        if seed is None:
            seed = int(time.time())
        self.rng = np.random.default_rng(seed)
        self.engine = TTREngine(rng=self.rng, num_players=2)
        self.engine.reset()
        self.lstm_state = None  # reset on new game — ai_fn will initialize
        self.last_ai_actions = []
        self.action_log = []
        # Human is player 0, AI is player 1
        self._run_ai_turns()

    def human_action(self, action: int):
        if self.engine is None or self.engine.done:
            return
        if self.engine.current_player != 0:
            return
        mask = self.engine.legal_actions()
        if not mask[action]:
            return
        self.action_log.append({"player": 0, "action": action, "desc": _describe_action(action)})
        self.engine.step(action)
        # After human turn (when FINISHED → next player), run AI
        self._run_ai_turns()

    def _run_ai_turns(self):
        """Run AI turns until it's human's turn or game over."""
        self.last_ai_actions = []
        while not self.engine.done and self.engine.current_player == 1:
            mask = self.engine.legal_actions()
            if not mask.any():
                break
            if self.ai_fn is not None:
                action, self.lstm_state = self.ai_fn(self.engine, 1, self.lstm_state)
            else:
                # Random
                action = int(self.rng.choice(np.where(mask)[0]))
            desc = _describe_action(action)
            self.last_ai_actions.append({"action": action, "desc": desc})
            self.action_log.append({"player": 1, "action": action, "desc": desc})
            self.engine.step(action)

    def get_state(self) -> dict:
        if self.engine is None:
            return {"started": False}

        e = self.engine
        p0 = e._players[0]
        p1 = e._players[1]

        # Decode actions for human
        mask = e.legal_actions() if not e.done and e.current_player == 0 else np.zeros(NUM_ACTIONS, dtype=bool)
        legal_actions = []
        for a in range(NUM_ACTIONS):
            if not mask[a]:
                continue
            legal_actions.append({"id": a, "desc": _describe_action(a)})

        # Routes info
        routes = []
        for rid in range(NUM_ROUTES):
            owner = int(e._route_owner[rid])
            routes.append({
                "id": rid,
                "city1": CITIES[ROUTE_CITY1[rid]],
                "city2": CITIES[ROUTE_CITY2[rid]],
                "color": COLOR_NAMES[ROUTE_COLOR[rid]],
                "length": ROUTE_LENGTH[rid],
                "points": ROUTE_POINTS_LIST[rid],
                "owner": owner,  # -1=unclaimed, 0=human, 1=AI
            })

        # Destinations
        dest_info = []
        for did in range(NUM_DESTINATIONS):
            status = "unclaimed"
            if did in p0.uncompleted_dest:
                status = "uncompleted"
            elif did in p0.completed_dest:
                status = "completed"
            dest_info.append({
                "id": did,
                "city1": CITIES[DEST_CITY1[did]],
                "city2": CITIES[DEST_CITY2[did]],
                "points": DEST_POINTS[did],
                "status": status,
            })

        # Available destinations for selection
        avail_dest = [{"id": did, "city1": CITIES[DEST_CITY1[did]],
                       "city2": CITIES[DEST_CITY2[did]], "points": DEST_POINTS[did]}
                      for did in e._dest_available]

        game_state_names = ["FIRST_ROUND", "PLAYING", "LAST_ROUND", "GAME_OVER"]
        turn_state_names = ["INIT", "SELECTING_DEST", "DRAWING_CARDS", "FINISHED"]

        return {
            "started": True,
            "done": e.done,
            "game_state": game_state_names[e._game_state],
            "turn_state": turn_state_names[e._turn_state],
            "current_player": e.current_player,
            "human": {
                "hand": {COLOR_NAMES[c]: int(p0.hand[c]) for c in range(NUM_COLORS)},
                "trains": p0.trains,
                "points": p0.points,
                "routes_claimed": len(p0.routes),
            },
            "ai": {
                "hand": {COLOR_NAMES[c]: int(p1.hand[c]) for c in range(NUM_COLORS)},
                "hand_size": int(p1.hand.sum()),
                "trains": p1.trains,
                "points": p1.points,
                "routes_claimed": len(p1.routes),
                "dest_count": len(p1.uncompleted_dest) + len(p1.completed_dest),
                "destinations": [
                    {"id": did, "city1": CITIES[DEST_CITY1[did]],
                     "city2": CITIES[DEST_CITY2[did]], "points": DEST_POINTS[did],
                     "status": "completed" if did in p1.completed_dest else "uncompleted"}
                    for did in list(p1.completed_dest) + list(p1.uncompleted_dest)
                ],
            },
            "visible_cards": {COLOR_NAMES[c]: int(e._visible[c]) for c in range(NUM_COLORS)},
            "deck_size": int(e._deck.sum()),
            "legal_actions": legal_actions,
            "routes": routes,
            "destinations": dest_info,
            "avail_dest": avail_dest,
            "scores": [p0.points, p1.points] if e.done else None,
            "ai_actions": self.last_ai_actions,
            "action_log": self.action_log[-20:],  # last 20 actions
        }


def _describe_action(a: int) -> str:
    if a == ACT_DRAW_RANDOM:
        return "Draw from deck"
    if ACT_DRAW_VISIBLE_BASE <= a <= ACT_DRAW_VISIBLE_BASE + 8:
        color = a - ACT_DRAW_VISIBLE_BASE
        return f"Draw visible {COLOR_NAMES[color]}"
    if a == ACT_DRAW_DEST:
        return "Draw destinations"
    if ACT_SELECT_DEST_BASE <= a < ACT_SELECT_DEST_BASE + NUM_DESTINATIONS:
        did = a - ACT_SELECT_DEST_BASE
        return f"Select: {CITIES[DEST_CITY1[did]]} → {CITIES[DEST_CITY2[did]]} ({DEST_POINTS[did]}pts)"
    if a == ACT_FINISH_SELECT:
        return "Finish selecting destinations"
    if ACT_CLAIM_ROUTE_BASE <= a < ACT_CLAIM_ROUTE_BASE + NUM_ROUTES:
        rid = a - ACT_CLAIM_ROUTE_BASE
        return f"Claim: {CITIES[ROUTE_CITY1[rid]]}–{CITIES[ROUTE_CITY2[rid]]} ({COLOR_NAMES[ROUTE_COLOR[rid]]}, len {ROUTE_LENGTH[rid]}, {ROUTE_POINTS_LIST[rid]}pts)"
    return f"Action {a}"


# ── Global session ───────────────────────────────────────────────────────────

session = GameSession()
_model_dir = None  # set in main()
_loaded_models = {}  # cache: filename -> ai_fn

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(_STATIC_DIR, "index.html", max_age=0)

@app.route("/api/new", methods=["POST"])
def new_game():
    session.new_game()
    return jsonify(session.get_state())

@app.route("/api/state")
def get_state():
    return jsonify(session.get_state())

@app.route("/api/action", methods=["POST"])
def do_action():
    data = request.get_json()
    action = data.get("action")
    if action is not None:
        session.human_action(int(action))
    return jsonify(session.get_state())

@app.route("/api/models")
def list_models():
    """List available model checkpoints."""
    models = [{"id": "random", "name": "Random"}]
    if _model_dir and os.path.isdir(_model_dir):
        for f in sorted(os.listdir(_model_dir)):
            if f.endswith(".pkt"):
                models.append({"id": f, "name": f.replace(".pkt", "")})
    return jsonify(models)


@app.route("/api/set_model", methods=["POST"])
def set_model():
    """Switch AI model. Accepts {"model": "latest.pkt"} or {"model": "random"}."""
    data = request.get_json()
    model_id = data.get("model", "random")

    if model_id == "random":
        session.ai_fn = None
        return jsonify({"status": "ok", "model": "random"})

    if _model_dir is None:
        return jsonify({"status": "error", "msg": "no model directory configured"}), 400

    path = os.path.join(_model_dir, model_id)
    if not os.path.isfile(path):
        return jsonify({"status": "error", "msg": f"model not found: {model_id}"}), 404

    if model_id not in _loaded_models:
        import torch
        from game.ttr.agent import TTRAgent
        from abstract import make_lstm_state
        from abstract.ppo_lstm import obs_to_tensor

        device = "cpu"
        agent = TTRAgent(num_players=2).to(device)
        agent.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        agent.eval()
        print(f"Loaded model from {path}")

        def ai_fn(engine, player_id, lstm_state, _agent=agent):
            obs = engine.encode(player_id)
            obs_t = obs_to_tensor(TTRObs(*[np.expand_dims(f, 0) for f in obs]), device)
            mask = engine.legal_actions()
            mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            if lstm_state is None:
                lstm_state = make_lstm_state(_agent.lstm_layers, 1, _agent.lstm_hidden, device)
            done_t = torch.zeros(1)
            with torch.no_grad():
                action, _, _, _, new_lstm_state = _agent.get_action_and_value(obs_t, mask_t, lstm_state, done_t)
            return int(action.item()), new_lstm_state

        _loaded_models[model_id] = ai_fn

    session.ai_fn = _loaded_models[model_id]
    return jsonify({"status": "ok", "model": model_id})


@app.route("/api/map_data")
def map_data():
    """Static map data: city coords, route geometry, destination info."""
    cities = []
    for i, name in enumerate(CITIES):
        lat, lon = CITY_COORDS[name]
        cities.append({"id": i, "name": name, "lat": lat, "lon": lon})
    routes = []
    for rid in range(NUM_ROUTES):
        routes.append({
            "id": rid,
            "city1": ROUTE_CITY1[rid], "city2": ROUTE_CITY2[rid],
            "color": COLOR_NAMES[ROUTE_COLOR[rid]],
            "length": ROUTE_LENGTH[rid],
            "points": ROUTE_POINTS_LIST[rid],
            "adjacent": ROUTE_ADJACENT[rid],
        })
    dests = []
    for did in range(NUM_DESTINATIONS):
        dests.append({
            "id": did,
            "city1": DEST_CITY1[did], "city2": DEST_CITY2[did],
            "points": DEST_POINTS[did],
        })
    return jsonify({"cities": cities, "routes": routes, "destinations": dests})



# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ticket to Ride server")
    parser.add_argument("--model-dir", default=None, help="Model directory for AI opponent")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global _model_dir
    _model_dir = args.model_dir

    if args.model_dir:
        # Auto-load latest.pkt if it exists
        path = os.path.join(args.model_dir, "latest.pkt")
        if os.path.isfile(path):
            import torch
            from game.ttr.agent import TTRAgent
            from abstract import make_lstm_state
            from abstract.ppo_lstm import obs_to_tensor

            device = "cpu"
            agent = TTRAgent(num_players=2).to(device)
            agent.load_state_dict(torch.load(path, map_location=device, weights_only=False))
            agent.eval()
            print(f"Loaded model from {path}")

            def ai_fn(engine, player_id, lstm_state):
                obs = engine.encode(player_id)
                obs_t = obs_to_tensor(TTRObs(*[np.expand_dims(f, 0) for f in obs]), device)
                mask = engine.legal_actions()
                mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
                if lstm_state is None:
                    lstm_state = make_lstm_state(agent.lstm_layers, 1, agent.lstm_hidden, device)
                done_t = torch.zeros(1)
                with torch.no_grad():
                    action, _, _, _, new_lstm_state = agent.get_action_and_value(obs_t, mask_t, lstm_state, done_t)
                return int(action.item()), new_lstm_state

            session.ai_fn = ai_fn
            _loaded_models["latest.pkt"] = ai_fn

    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
