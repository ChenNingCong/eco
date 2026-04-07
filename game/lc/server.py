"""
server.py — Flask server to play Lost Cities against AI.

Usage:
    python -m game.lc.server                        # random opponent
    python -m game.lc.server --model-dir model/lc   # with trained agent
    python -m game.lc.server --port 5002
"""

import argparse
import os
import time

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from game.lc.engine import (
    LCEngine, LCObs, NUM_COLORS, NUM_UNIQUE_VALUES, NUM_CARD_IDS,
    NUM_ACTIONS, NUM_DECOMPOSED_ACTIONS, NUM_PLAY_ACTIONS,
    PHASE_PLAY, PHASE_DRAW,
    COLOR_NAMES, CARD_VALUES,
    card_id_to_color_value, color_value_to_card_id,
    encode_action, decode_action,
    encode_play_action, decode_play_action,
    encode_draw_action, decode_draw_action,
)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=_STATIC_DIR)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _card_id_to_tuple(card_id: int) -> list:
    """Convert card_id to [color_idx, value] for frontend."""
    color, val_idx = card_id_to_color_value(card_id)
    return [color, CARD_VALUES[val_idx]]


def _hand_array_to_list(hand: np.ndarray) -> list:
    """Convert count-based hand array to list of [color, value] cards."""
    cards = []
    for card_id in range(NUM_CARD_IDS):
        count = int(hand[card_id])
        for _ in range(count):
            cards.append(_card_id_to_tuple(card_id))
    return cards


def _expedition_to_cards(val_idx_list: list, color: int) -> list:
    """Convert expedition value_idx list to [[color, value], ...] for frontend."""
    return [[color, CARD_VALUES[v]] for v in val_idx_list]


# ── Game session ────────────────────────────────────────────────────────────

class GameSession:
    def __init__(self, ai_fn=None, max_lanes=5, decompose_actions=False, max_discard_draws=50):
        self.ai_fn = ai_fn  # callable(engine, player_id, lstm_state) -> (action, new_lstm_state)
        self.engine = None
        self.rng = None
        self.lstm_state = None
        self.max_lanes = max_lanes
        self.decompose_actions = decompose_actions
        self.max_discard_draws = max_discard_draws
        self.autoplay_log = []  # log of actions for autoplay replay

    def new_game(self, seed=None):
        if seed is None:
            seed = int(time.time())
        self.rng = np.random.default_rng(seed)
        self.engine = LCEngine(rng=self.rng, max_lanes=self.max_lanes,
                               decompose_actions=self.decompose_actions,
                               max_discard_draws=self.max_discard_draws)
        self.engine.reset()
        self.lstm_state = None
        self.autoplay_log = []
        # If AI goes first (player 1 starts), run its turns
        self._run_ai_turns()

    def human_action(self, card_index: int, action_type: str, draw_source: int):
        """Execute human (player 0) action.
        card_index: index in the hand list
        action_type: 'E' for expedition, 'D' for discard
        draw_source: 0=deck, 1-5=discard piles
        """
        e = self.engine
        if e is None or e.done or e.current_player != 0:
            return

        # Convert hand-index to card_id
        hand_list = _hand_array_to_list(e._hands[0])
        if card_index >= len(hand_list):
            return
        card_color, card_value = hand_list[card_index]
        if card_value == 0:
            val_idx = 0
        else:
            val_idx = card_value - 1
        card_id = color_value_to_card_id(card_color, val_idx)

        at = 0 if action_type == 'E' else 1

        if self.decompose_actions:
            # Two-phase: play then draw
            play_action = encode_play_action(card_id, at)
            mask = e.legal_actions()
            if not mask[play_action]:
                return
            e.step(play_action)

            draw_action = encode_draw_action(draw_source)
            mask = e.legal_actions()
            if not mask[draw_action]:
                return
            e.step(draw_action)
        else:
            action = encode_action(card_id, at, draw_source)
            mask = e.legal_actions()
            if not mask[action]:
                return
            e.step(action)

        self._run_ai_turns()

    def _run_ai_turns(self):
        e = self.engine
        while not e.done and e.current_player == 1:
            mask = e.legal_actions()
            if not mask.any():
                break
            if self.ai_fn is not None:
                action, self.lstm_state = self.ai_fn(e, 1, self.lstm_state)
            else:
                action = int(self.rng.choice(np.where(mask)[0]))
            e.step(action)
            # In decomposed mode, AI also needs the draw step
            if self.decompose_actions and not e.done and e.current_player == 1 and e._phase == PHASE_DRAW:
                mask = e.legal_actions()
                if not mask.any():
                    break
                if self.ai_fn is not None:
                    action, self.lstm_state = self.ai_fn(e, 1, self.lstm_state)
                else:
                    action = int(self.rng.choice(np.where(mask)[0]))
                e.step(action)

    def get_state(self, player_id: int = 0) -> dict:
        if self.engine is None:
            return {"started": False, "game_over": True}

        e = self.engine
        scores = e.compute_scores()

        # Build hand as list of [color, value] tuples
        hand = _hand_array_to_list(e._hands[player_id])

        # Build expeditions: [player][color] -> list of [color, value]
        expeditions = []
        for p in range(2):
            player_exps = []
            for c in range(NUM_COLORS):
                player_exps.append(_expedition_to_cards(e._expeditions[p][c], c))
            expeditions.append(player_exps)

        # Discard piles: top card or None
        discard_top = []
        full_discard = []
        for c in range(NUM_COLORS):
            pile = e._discard_piles[c]
            if pile:
                discard_top.append(_card_id_to_tuple(pile[-1]))
            else:
                discard_top.append(None)
            full_discard.append([_card_id_to_tuple(cid) for cid in pile])

        return {
            "player_id": player_id,
            "current_player": e.current_player,
            "game_over": e.done,
            "scores": [float(scores[0]), float(scores[1])],
            "deck_size": len(e._deck),
            "hand": hand,
            "opponent_hand_size": int(e._hands[1 - player_id].sum()),
            "expeditions": expeditions,
            "discard_piles": discard_top,
            "full_discard_piles": full_discard,
            "color_names": COLOR_NAMES,
        }


# ── Global session ──────────────────────────────────────────────────────────

session = GameSession()
_model_dirs = []
_dqn_dirs = []
_dqn_hidden_dim = 256
_ppo_hidden_dim = 256
_dqn_model_path = None  # legacy single file
_max_lanes = 5
_loaded_models = {}


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(_STATIC_DIR, "lost-cities.html", max_age=0)


@app.route("/game/reset")
def reset_game():
    session.new_game()
    return jsonify({"message": "Game reset successfully. New game started."})


@app.route("/game/state/<int:player_id>")
def get_state(player_id: int):
    return jsonify(session.get_state(player_id))


@app.route("/game/valid_actions/<int:player_id>")
def get_valid_actions(player_id: int):
    e = session.engine
    if e is None or e.current_player != player_id:
        return jsonify({"plays": [], "draws": []})

    hand = _hand_array_to_list(e._hands[player_id])
    valid_plays = []
    for i, (color, value) in enumerate(hand):
        val_idx = 0 if value == 0 else value - 1
        card_id = color_value_to_card_id(color, val_idx)
        if e._is_valid_expedition_play(player_id, card_id):
            valid_plays.append((i, 'E', color))
        valid_plays.append((i, 'D', color))

    valid_draws = [0]
    for c in range(NUM_COLORS):
        if e._discard_piles[c]:
            valid_draws.append(c + 1)

    return jsonify({"plays": valid_plays, "draws": valid_draws})


@app.route("/game/play", methods=["POST"])
def play_action():
    data = request.get_json()
    card_index = data.get("card_index")
    action_type = data.get("action_type")
    draw_source = data.get("draw_source")

    if card_index is None or action_type is None or draw_source is None:
        return jsonify({"detail": "Missing action parameters."}), 400

    session.human_action(int(card_index), action_type, int(draw_source))
    state = session.get_state(0)
    return jsonify({"message": "Move played.", "state": state})


@app.route("/game/autoplay")
def autoplay():
    """Run a full AI vs AI game and return step-by-step states."""
    session.new_game()
    e = session.engine
    ai_fn = session.ai_fn
    if ai_fn is None:
        return jsonify({"detail": "No AI model loaded."}), 400

    # We need two separate LSTM states for the two players
    lstm_states = [None, None]
    steps = [session.get_state(0)]  # initial state from player 0 perspective

    while not e.done:
        p = e.current_player
        if session.decompose_actions:
            # Play phase
            play_act, lstm_states[p] = ai_fn(e, p, lstm_states[p])
            card_id, action_type = decode_play_action(play_act)
            color, val_idx = card_id_to_color_value(card_id)
            e.step(play_act)
            if e.done:
                break
            # Draw phase
            draw_act, lstm_states[p] = ai_fn(e, p, lstm_states[p])
            draw_source = decode_draw_action(draw_act)
            e.step(draw_act)
        else:
            action, lstm_states[p] = ai_fn(e, p, lstm_states[p])
            card_id, action_type, draw_source = decode_action(action)
            color, val_idx = card_id_to_color_value(card_id)
            e.step(action)
        action_desc = {
            "player": p,
            "card": [color, CARD_VALUES[val_idx]],
            "action": "expedition" if action_type == 0 else "discard",
            "draw": "deck" if draw_source == 0 else f"discard_{COLOR_NAMES[draw_source - 1]}",
        }
        state = session.get_state(0)
        state["last_action"] = action_desc
        steps.append(state)

    return jsonify({"steps": steps, "total_steps": len(steps)})


@app.route("/game/opponent")
def list_opponents():
    """Return structured runs with checkpoints for two-dropdown UI."""
    runs = []

    def _sort_key(fname):
        """Sort checkpoints: 'latest' first, then by numeric step, then alpha."""
        if fname.startswith("latest"):
            return (-1, 0, fname)
        import re
        nums = re.findall(r'\d+', fname)
        step = int(nums[-1]) if nums else 0
        return (0, step, fname)

    # DQN runs from --dqn-dir
    for dqn_dir in _dqn_dirs:
        if os.path.isdir(dqn_dir):
            run_name = os.path.basename(dqn_dir)
            ckpts = []
            for f in sorted(os.listdir(dqn_dir), key=_sort_key):
                if f.endswith(".pt"):
                    path = os.path.join(dqn_dir, f)
                    ckpts.append({"id": path, "name": f})
            if ckpts:
                runs.append({"name": f"[DQN] {run_name}", "type": "dqn", "checkpoints": ckpts})

    # Legacy single DQN model
    if _dqn_model_path and not _dqn_dirs:
        runs.append({"name": "[DQN] legacy", "type": "dqn",
                      "checkpoints": [{"id": _dqn_model_path, "name": os.path.basename(_dqn_model_path)}]})

    # PPO runs from --model-dir (supports multiple dirs)
    for mdir in _model_dirs:
        if os.path.isdir(mdir):
            run_name = os.path.basename(mdir)
            ckpts = []
            for f in sorted(os.listdir(mdir), key=_sort_key):
                if f.endswith(".pkt"):
                    path = os.path.join(mdir, f)
                    ckpts.append({"id": path, "name": f})
            if ckpts:
                runs.append({"name": f"[PPO] {run_name}", "type": "ppo", "checkpoints": ckpts})

    return jsonify({"runs": runs})


@app.route("/game/set_opponent", methods=["POST"])
def set_opponent():
    data = request.get_json()
    model_name = data.get("path", "")

    # Resolve path: try name directly, then from each model_dir
    path = model_name if os.path.isfile(model_name) else None
    if path is None:
        for mdir in _model_dirs:
            candidate = os.path.join(mdir, model_name)
            if os.path.isfile(candidate):
                path = candidate
                break
    if path is None:
        return jsonify({"detail": f"Model not found: {model_name}"}), 404

    if path in _loaded_models:
        session.ai_fn = _loaded_models[path]
        return jsonify({"message": f"Opponent set to {model_name}"})

    import torch

    # DQN model (.pt)
    if path.endswith(".pt"):
        from game.lc.dqn_agent import LCQNetwork
        from abstract.dqn import _flatten_obs_batch

        device = "cpu"
        q_net = LCQNetwork(hidden_dim=_dqn_hidden_dim).to(device)
        q_net.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        q_net.eval()
        print(f"Loaded DQN model from {path}")

        def ai_fn(engine, player_id, lstm_state, _qnet=q_net):
            obs = engine.encode(player_id)
            obs_batch = LCObs(*[np.expand_dims(f, 0) for f in obs])
            obs_flat = _flatten_obs_batch(obs_batch)
            obs_t = torch.as_tensor(obs_flat, device="cpu")
            mask = engine.legal_actions()
            mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                q_vals = _qnet(obs_t)
                q_vals[~mask_t] = -1e8
                action = q_vals.argmax(dim=1)
            return int(action.item()), None

        _loaded_models[path] = ai_fn

    # PPO model (.pkt)
    else:
        from game.lc.agent import LCAgent
        from abstract import make_lstm_state
        from abstract.ppo_lstm import obs_to_tensor

        device = "cpu"
        agent = LCAgent(lstm_hidden=128, hidden_dim=_ppo_hidden_dim,
                        decompose_actions=session.decompose_actions).to(device)
        agent.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        agent.eval()
        print(f"Loaded PPO model from {path}")

        def ai_fn(engine, player_id, lstm_state, _agent=agent):
            obs = engine.encode(player_id)
            obs_t = obs_to_tensor(LCObs(*[np.expand_dims(f, 0) for f in obs]), device)
            mask = engine.legal_actions()
            mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            if lstm_state is None:
                lstm_state = make_lstm_state(_agent.lstm_layers, 1, _agent.lstm_hidden, device)
            done_t = torch.zeros(1)
            with torch.no_grad():
                action, _, _, _, new_lstm_state = _agent.get_action_and_value(
                    obs_t, mask_t, lstm_state, done_t)
            return int(action.item()), new_lstm_state

        _loaded_models[path] = ai_fn

    session.ai_fn = _loaded_models[path]
    # Match training conditions for dd5 models
    if "dd5" in path:
        session.max_discard_draws = 5
    else:
        session.max_discard_draws = 50
    return jsonify({"message": f"Opponent set to {model_name}"})


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lost Cities server")
    parser.add_argument("--model-dir", default=None, nargs="*",
                        help="PPO model directories to scan for .pkt checkpoints")
    parser.add_argument("--dqn-dir", default=None, nargs="*",
                        help="DQN model directories to scan for .pt checkpoints (e.g. model/lc_dqn_dense model/lc_dqn_h512)")
    parser.add_argument("--dqn-model", default=None, help="Path to a single DQN .pt checkpoint (legacy)")
    parser.add_argument("--dqn-hidden-dim", type=int, default=256, help="DQN hidden dim (must match checkpoint)")
    parser.add_argument("--ppo-hidden-dim", type=int, default=256, help="PPO hidden dim (must match checkpoint)")
    parser.add_argument("--max-lanes", type=int, default=5, help="Max expedition lanes (default 5)")
    parser.add_argument("--decompose-actions", action="store_true", help="Use decomposed action space (106 instead of 600)")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global _model_dirs, _dqn_dirs, _dqn_hidden_dim, _ppo_hidden_dim, _dqn_model_path, _max_lanes, session
    _model_dirs = args.model_dir or []
    _dqn_dirs = args.dqn_dir or []
    _dqn_hidden_dim = args.dqn_hidden_dim
    _ppo_hidden_dim = args.ppo_hidden_dim
    _max_lanes = args.max_lanes
    session = GameSession(max_lanes=_max_lanes, decompose_actions=args.decompose_actions)

    if _model_dirs:
        # Auto-load latest checkpoint from first model dir
        latest = None
        first_dir = _model_dirs[0]
        if os.path.isdir(first_dir):
            pkts = sorted([f for f in os.listdir(first_dir) if f.endswith(".pkt")])
            if pkts:
                latest = pkts[-1]
        if latest:
            path = os.path.join(first_dir, latest)
            import torch
            from game.lc.agent import LCAgent
            from abstract import make_lstm_state
            from abstract.ppo_lstm import obs_to_tensor

            device = "cpu"
            agent = LCAgent(lstm_hidden=128, hidden_dim=args.ppo_hidden_dim,
                            decompose_actions=args.decompose_actions).to(device)
            agent.load_state_dict(torch.load(path, map_location=device, weights_only=False))
            agent.eval()
            print(f"Loaded model from {path} (hidden_dim={args.ppo_hidden_dim})")

            def ai_fn(engine, player_id, lstm_state):
                obs = engine.encode(player_id)
                obs_t = obs_to_tensor(LCObs(*[np.expand_dims(f, 0) for f in obs]), device)
                mask = engine.legal_actions()
                mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
                if lstm_state is None:
                    lstm_state = make_lstm_state(agent.lstm_layers, 1, agent.lstm_hidden, device)
                done_t = torch.zeros(1)
                with torch.no_grad():
                    action, _, _, _, new_lstm_state = agent.get_action_and_value(
                        obs_t, mask_t, lstm_state, done_t)
                return int(action.item()), new_lstm_state

            session.ai_fn = ai_fn
            _loaded_models[path] = ai_fn

    if args.dqn_model and os.path.isfile(args.dqn_model):
        import torch
        from game.lc.dqn_agent import LCQNetwork
        from abstract.dqn import _flatten_obs_batch

        device = "cpu"
        q_net = LCQNetwork(hidden_dim=args.dqn_hidden_dim).to(device)
        q_net.load_state_dict(torch.load(args.dqn_model, map_location=device, weights_only=False))
        q_net.eval()
        print(f"Loaded DQN model from {args.dqn_model} (hidden_dim={args.dqn_hidden_dim})")
        _dqn_model_path = args.dqn_model

        def ai_fn(engine, player_id, lstm_state, _qnet=q_net):
            obs = engine.encode(player_id)
            obs_batch = LCObs(*[np.expand_dims(f, 0) for f in obs])
            obs_flat = _flatten_obs_batch(obs_batch)
            obs_t = torch.as_tensor(obs_flat, device=device)
            mask = engine.legal_actions()
            mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                q_vals = _qnet(obs_t)
                q_vals[~mask_t] = -1e8
                action = q_vals.argmax(dim=1)
            return int(action.item()), None

        session.ai_fn = ai_fn
        _loaded_models[args.dqn_model] = ai_fn

    session.new_game()
    print(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
