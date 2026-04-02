"""
Analyze R-öko checkpoint behavior: run games between checkpoints and measure
detailed metrics to understand strategy cycling in self-play.

Usage:
    python eco_analyze.py --model-dir model --num-games 200
"""

import argparse
import glob
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from eco_env import (EcoEnv, EcoState, NUM_COLORS, NUM_ACTIONS, NUM_PLAY_ACTIONS,
                     PHASE_PLAY, PHASE_DISCARD, decode_play, decode_discard,
                     HAND_LIMIT, _STACK)


class RandomAgent:
    def act(self, mask):
        return int(np.random.choice(np.where(mask)[0]))


class CheckpointAgent:
    def __init__(self, path, num_players, device="cuda"):
        import torch
        from eco_ppo import EcoAgent, obs_to_tensor
        self.device = torch.device(device)
        self.agent = EcoAgent(num_players=num_players).to(self.device)
        self.agent.load_state_dict(torch.load(path, map_location=self.device))
        self.agent.eval()
        self.num_players = num_players
        self._obs_to_tensor = obs_to_tensor

    def act_with_obs(self, env, seat):
        """Get action using the observation encoder."""
        import torch
        from eco_obs_encoder import SinglePlayerEcoEnv, EcoPyTreeObs
        mask = env.legal_actions()
        wrapper = SinglePlayerEcoEnv.__new__(SinglePlayerEcoEnv)
        wrapper.env = env
        wrapper._num_players = self.num_players
        wrapper._seat = seat
        obs = wrapper._encode_for(seat)
        obs_t = self._obs_to_tensor(
            EcoPyTreeObs(*[np.expand_dims(f, 0) for f in obs]),
            self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(obs_t, mask_t)
        return int(action.item())


def run_game(env, agents, num_players):
    """Run a single game. Returns per-player metrics dict."""
    state = env.reset()
    metrics = {p: {
        "play_values": [],      # value of each play action
        "play_colors": [],      # color played
        "discards": 0,          # number of discard actions
        "tokens_claimed": 0,    # number of factory tokens claimed
        "token_values": [],     # values of claimed tokens
        "waste_picked_up": 0,   # total waste cards picked up
        "turns": 0,
    } for p in range(num_players)}

    while not state.done:
        seat = int(state.current_player)
        mask = env.legal_actions()

        # Record pre-action state
        pre_stacks = [len(state.factory_stacks[c]) for c in range(NUM_COLORS)]
        pre_waste = state.waste_side.copy()

        # Get action
        agent = agents[seat]
        if isinstance(agent, RandomAgent):
            action = agent.act(mask)
        else:
            action = agent.act_with_obs(env, seat)

        state, _, done, _ = env.step(action)
        metrics[seat]["turns"] += 1

        if action < NUM_PLAY_ACTIONS:
            color, ns, nd = decode_play(action)
            value = ns + nd * 2
            metrics[seat]["play_values"].append(value)
            metrics[seat]["play_colors"].append(color)

            # Check if token was claimed (stack got shorter)
            post_stacks = [len(state.factory_stacks[c]) for c in range(NUM_COLORS)]
            if post_stacks[color] < pre_stacks[color]:
                metrics[seat]["tokens_claimed"] += 1
                # The token value was the top of the old stack
                stack_template = _STACK[num_players]
                consumed_before = len(stack_template) - pre_stacks[color]
                if consumed_before < len(stack_template):
                    metrics[seat]["token_values"].append(stack_template[consumed_before])

            # Waste pickup: compare pre vs post waste for that factory color
            pre_waste_total = int(pre_waste[color].sum())
            post_waste_total = int(state.waste_side[color].sum())
            if pre_waste_total > 0 and post_waste_total < pre_waste_total:
                metrics[seat]["waste_picked_up"] += pre_waste_total
        else:
            metrics[seat]["discards"] += 1

    scores = env.compute_scores(state)
    for p in range(num_players):
        metrics[p]["score"] = float(scores[p])
        metrics[p]["penalty_count"] = int(state.penalty_pile[p].sum())
        # Count collected tokens per color
        total_tokens = sum(len(state.collected[p][c]) for c in range(NUM_COLORS))
        metrics[p]["total_tokens"] = total_tokens
        # Count colors with 2+ tokens (scoring colors)
        scoring_colors = sum(1 for c in range(NUM_COLORS) if len(state.collected[p][c]) >= 2)
        metrics[p]["scoring_colors"] = scoring_colors

    winner = int(np.argmax(scores))
    return metrics, winner


def aggregate_metrics(all_metrics, num_players):
    """Aggregate per-game metrics into summary statistics."""
    agg = {}
    for p in range(num_players):
        player_data = [m[p] for m in all_metrics]
        avg_score = np.mean([d["score"] for d in player_data])
        avg_penalty = np.mean([d["penalty_count"] for d in player_data])
        avg_tokens = np.mean([d["total_tokens"] for d in player_data])
        avg_scoring = np.mean([d["scoring_colors"] for d in player_data])
        avg_discards = np.mean([d["discards"] for d in player_data])
        avg_waste = np.mean([d["waste_picked_up"] for d in player_data])
        avg_play_val = np.mean([v for d in player_data for v in d["play_values"]]) if any(d["play_values"] for d in player_data) else 0
        avg_turns = np.mean([d["turns"] for d in player_data])

        # Play color distribution
        color_counts = np.zeros(NUM_COLORS)
        for d in player_data:
            for c in d["play_colors"]:
                color_counts[c] += 1
        total_plays = color_counts.sum()
        color_dist = color_counts / max(total_plays, 1)

        agg[p] = {
            "avg_score": avg_score,
            "avg_penalty": avg_penalty,
            "avg_tokens": avg_tokens,
            "avg_scoring_colors": avg_scoring,
            "avg_discards": avg_discards,
            "avg_waste_picked_up": avg_waste,
            "avg_play_value": avg_play_val,
            "avg_turns": avg_turns,
            "color_distribution": color_dist,
        }
    return agg


def get_checkpoint_paths(model_dir):
    """Get all checkpoint paths sorted by step number."""
    files = glob.glob(os.path.join(model_dir, "eco_*.pkt"))
    files = [f for f in files if "latest" not in os.path.basename(f)]

    def step_num(f):
        base = os.path.basename(f).replace("eco_", "").replace(".pkt", "")
        try:
            return int(base)
        except ValueError:
            return 0

    files.sort(key=step_num)
    return files


def select_checkpoints(files, max_count=8):
    """Select checkpoints with exponential spacing."""
    if len(files) <= max_count:
        return files
    # Always include first and last, then exponentially spaced in between
    indices = set([0, len(files) - 1])
    idx = len(files) - 1
    gap = 1
    while idx >= 0 and len(indices) < max_count:
        indices.add(idx)
        idx -= gap
        gap *= 2
    return [files[i] for i in sorted(indices)]


def main():
    parser = argparse.ArgumentParser(description="Analyze R-öko checkpoint behaviors")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--num-players", type=int, default=3)
    parser.add_argument("--max-checkpoints", type=int, default=8,
                        help="Max checkpoints to test (exponential spacing)")
    parser.add_argument("--step-gap", type=int, default=0,
                        help="If >0, select checkpoints at this step interval (nearest available)")
    args = parser.parse_args()

    all_files = get_checkpoint_paths(args.model_dir)
    if not all_files:
        print("No checkpoints found in", args.model_dir)
        return

    if args.step_gap > 0:
        # Select checkpoints at fixed step intervals (nearest available)
        def _step(f):
            base = os.path.basename(f).replace("eco_", "").replace(".pkt", "")
            try: return int(base)
            except ValueError: return 0
        all_steps = [_step(f) for f in all_files]
        max_step = max(all_steps)
        targets = list(range(0, max_step + 1, args.step_gap))
        if max_step not in targets:
            targets.append(max_step)
        selected = []
        used = set()
        for t in targets:
            best_idx = min(range(len(all_steps)), key=lambda i: abs(all_steps[i] - t))
            if best_idx not in used:
                selected.append(all_files[best_idx])
                used.add(best_idx)
    else:
        selected = select_checkpoints(all_files, args.max_checkpoints)
    print(f"Found {len(all_files)} checkpoints, selected {len(selected)}:")
    for f in selected:
        step = os.path.basename(f).replace("eco_", "").replace(".pkt", "")
        print(f"  Step {step}")

    np_ = args.num_players

    # ── Self-play analysis: each checkpoint plays itself ──
    print("\n" + "=" * 80)
    print("SELF-PLAY ANALYSIS (each checkpoint plays 3 copies of itself)")
    print("=" * 80)

    for ckpt_path in selected:
        step = os.path.basename(ckpt_path).replace("eco_", "").replace(".pkt", "")
        print(f"\n--- Step {step} (self-play) ---")

        agent = CheckpointAgent(ckpt_path, np_)
        agents = [agent] * np_

        all_metrics = []
        winners = []
        for _ in tqdm(range(args.num_games), leave=False):
            env = EcoEnv(num_players=np_)
            m, w = run_game(env, agents, np_)
            all_metrics.append(m)
            winners.append(w)

        agg = aggregate_metrics(all_metrics, np_)

        # Seat-wise win rates
        seat_wins = [0] * np_
        for w in winners:
            seat_wins[w] += 1
        seat_wr = [seat_wins[p] / args.num_games * 100 for p in range(np_)]

        # Since all agents are identical, average across all seats
        avg = {k: np.mean([agg[p][k] for p in range(np_)]) for k in agg[0] if k != "color_distribution"}
        color_dist = np.mean([agg[p]["color_distribution"] for p in range(np_)], axis=0)

        print(f"  Seat win rates:   " + "  ".join(f"P{p}={seat_wr[p]:.0f}%" for p in range(np_)))
        print(f"  Avg score:        " + "  ".join(f"P{p}={agg[p]['avg_score']:.1f}" for p in range(np_)))
        print(f"  Avg penalty:      " + "  ".join(f"P{p}={agg[p]['avg_penalty']:.1f}" for p in range(np_)))
        print(f"  Avg tokens:       {avg['avg_tokens']:.1f}")
        print(f"  Avg scoring cols: {avg['avg_scoring_colors']:.1f}")
        print(f"  Avg discards:     {avg['avg_discards']:.1f}")
        print(f"  Avg waste pickup: {avg['avg_waste_picked_up']:.1f}")
        print(f"  Avg play value:   {avg['avg_play_value']:.2f}")
        print(f"  Avg turns/player: {avg['avg_turns']:.1f}")
        print(f"  Color dist:       Glass={color_dist[0]:.2f} Paper={color_dist[1]:.2f} Plastic={color_dist[2]:.2f} Tin={color_dist[3]:.2f}")

    # ── Cross-play analysis: different checkpoints against each other ──
    if len(selected) >= 2:
        print("\n" + "=" * 80)
        print("CROSS-PLAY ANALYSIS (1 copy of ckpt_A vs 2 copies of ckpt_B)")
        print("=" * 80)

        random_agent = RandomAgent()

        # First: each checkpoint vs random
        print("\n--- vs Random ---")
        for ckpt_path in selected:
            step = os.path.basename(ckpt_path).replace("eco_", "").replace(".pkt", "")
            agent = CheckpointAgent(ckpt_path, np_)
            # Agent at seat 0, random at seats 1,2
            agents = [agent] + [random_agent] * (np_ - 1)
            wins = 0
            all_metrics = []
            for _ in tqdm(range(args.num_games), leave=False):
                env = EcoEnv(num_players=np_)
                m, w = run_game(env, agents, np_)
                all_metrics.append(m)
                if w == 0:
                    wins += 1

            agg = aggregate_metrics(all_metrics, np_)
            print(f"  Step {step:>8s}: win={wins/args.num_games*100:.0f}%  "
                  f"score={agg[0]['avg_score']:.1f}  "
                  f"penalty={agg[0]['avg_penalty']:.1f}  "
                  f"tokens={agg[0]['avg_tokens']:.1f}  "
                  f"scoring_cols={agg[0]['avg_scoring_colors']:.1f}")

        # Cross-checkpoint matchups
        print("\n--- Checkpoint A (seat 0) vs Checkpoint B (seats 1,2) ---")
        ab_label = "A \\ B"
        header = f"{ab_label:>12s}"
        for ckpt_b in selected:
            step_b = os.path.basename(ckpt_b).replace("eco_", "").replace(".pkt", "")
            header += f"  {step_b:>8s}"
        print(header)

        for ckpt_a in selected:
            step_a = os.path.basename(ckpt_a).replace("eco_", "").replace(".pkt", "")
            agent_a = CheckpointAgent(ckpt_a, np_)
            row = f"{step_a:>12s}"

            for ckpt_b in selected:
                if ckpt_a == ckpt_b:
                    row += f"  {'---':>8s}"
                    continue

                agent_b = CheckpointAgent(ckpt_b, np_)
                agents = [agent_a] + [agent_b] * (np_ - 1)
                wins = 0
                for _ in tqdm(range(args.num_games), leave=False):
                    env = EcoEnv(num_players=np_)
                    _, w = run_game(env, agents, np_)
                    if w == 0:
                        wins += 1

                wr = wins / args.num_games * 100
                row += f"  {wr:>7.0f}%"

            print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
