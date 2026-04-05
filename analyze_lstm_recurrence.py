"""
Test whether the LSTM actually uses its recurrent state.

Three ablation modes:
1. Zero weight_hh: kill gate conditioning on h_{t-1}, but cell still accumulates
2. Reset state every step: kill ALL temporal memory
3. Zero LSTM output: keep LSTM running but feed zeros to trunk

Also analyzes:
- Gate activations during a real game
- How much h changes with vs without history (same input)
- Whether the trunk's policy changes when LSTM output is zeroed
"""
import torch
import numpy as np
import argparse

from eco_ppo_lstm import EcoAgentFFLSTM, make_lstm_state, obs_to_tensor
from eco_obs_encoder import SinglePlayerEcoEnv, EcoPyTreeObs
from eco_env import EcoEnv


class RandomAgent:
    def act(self, mask):
        legal = np.where(mask)[0]
        return np.random.choice(legal)


def make_obs(env, seat, num_players):
    """Build observation tensor for a given seat."""
    wrapper = SinglePlayerEcoEnv.__new__(SinglePlayerEcoEnv)
    wrapper.env = env
    wrapper._num_players = num_players
    wrapper._seat = seat
    wrapper._relative_seat = True
    obs_raw = wrapper._encode_for(seat)
    obs = EcoPyTreeObs(*[np.expand_dims(f, 0) for f in obs_raw])
    return obs_to_tensor(obs, 'cpu')


def get_action(ag, env, seat, h_state, num_players=3, reset_state=False):
    obs_t = make_obs(env, seat, num_players)
    mask = env.legal_actions()
    mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
    done_t = torch.zeros(1)
    if reset_state:
        h_state = make_lstm_state(1, 1, ag.lstm_hidden, 'cpu')
    with torch.no_grad():
        action, _, _, _, new_h = ag.get_action_and_value(obs_t, mask_t, h_state, done_t)
    return int(action.item()), new_h


def play_vs_random(ag, num_games=500, num_players=3, reset_state=False):
    wins = 0
    random_ag = RandomAgent()
    for g in range(num_games):
        env = EcoEnv(num_players=num_players, seed=g)
        state = env.reset()
        h_state = make_lstm_state(1, 1, ag.lstm_hidden, 'cpu')

        while not state.done:
            seat = int(state.current_player)
            mask = env.legal_actions()
            if seat == 0:
                action, h_state = get_action(ag, env, seat, h_state,
                                             num_players, reset_state=reset_state)
            else:
                action = random_ag.act(mask)
            state, _, _, _ = env.step(action)

        scores = env.compute_scores()
        if scores[0] == max(scores):
            wins += 1

    return wins / num_games


# ── Analysis functions ────────────────────────────────────────────────────


def analyze_history_dependence(agent, num_players=3, seed=42):
    """Compare LSTM output h with vs without accumulated history for the same input."""
    env = EcoEnv(num_players=num_players, seed=seed)
    state = env.reset()
    h = torch.zeros(1, 1, agent.lstm_hidden)
    c = torch.zeros(1, 1, agent.lstm_hidden)

    steps = []
    with torch.no_grad():
        while not state.done:
            seat = int(state.current_player)
            if seat == 0:
                obs_t = make_obs(env, seat, num_players)
                shared = agent._shared_encode(obs_t)

                # With accumulated state
                _, (h, c) = agent.lstm(shared.unsqueeze(0), (h, c))
                h_with = h.squeeze().clone()

                # Same input, fresh state
                h0 = torch.zeros(1, 1, agent.lstm_hidden)
                c0 = torch.zeros(1, 1, agent.lstm_hidden)
                _, (h_fresh, _) = agent.lstm(shared.unsqueeze(0), (h0, c0))
                h_without = h_fresh.squeeze().clone()

                cos = torch.nn.functional.cosine_similarity(
                    h_with.unsqueeze(0), h_without.unsqueeze(0)).item()
                diff_norm = (h_with - h_without).norm().item()

                steps.append({
                    'c_norm': c.norm().item(),
                    'h_norm': h_with.norm().item(),
                    'h_fresh_norm': h_without.norm().item(),
                    'cos_sim': cos,
                    'diff_ratio': diff_norm / (h_with.norm().item() + 1e-8),
                })

            mask = env.legal_actions()
            state, _, _, _ = env.step(np.random.choice(np.where(mask)[0]))

    return steps


def analyze_trunk_sensitivity(agent, num_players=3, seed=42):
    """Check if trunk output/policy changes when LSTM output is zeroed."""
    env = EcoEnv(num_players=num_players, seed=seed)
    state = env.reset()
    h = torch.zeros(1, 1, agent.lstm_hidden)
    c = torch.zeros(1, 1, agent.lstm_hidden)

    steps = []
    with torch.no_grad():
        while not state.done:
            seat = int(state.current_player)
            if seat == 0:
                obs_t = make_obs(env, seat, num_players)
                shared = agent._shared_encode(obs_t)
                _, (h, c) = agent.lstm(shared.unsqueeze(0), (h, c))
                lstm_out = h.squeeze(0)  # (1, hidden)

                # Normal: concat(shared, lstm_out)
                combined_normal = torch.cat([shared, lstm_out], dim=-1)
                # Zeroed: concat(shared, zeros)
                combined_zero = torch.cat([shared, torch.zeros_like(lstm_out)], dim=-1)

                logits_normal = agent.actor_head(agent.actor_trunk(combined_normal))
                logits_zero = agent.actor_head(agent.actor_trunk(combined_zero))

                val_normal = agent.critic_head(agent.critic_trunk(combined_normal)).item()
                val_zero = agent.critic_head(agent.critic_trunk(combined_zero)).item()

                p_normal = torch.softmax(logits_normal, -1)
                p_zero = torch.softmax(logits_zero, -1)
                tv = (p_normal - p_zero).abs().sum().item() / 2

                steps.append({
                    'action_same': logits_normal.argmax().item() == logits_zero.argmax().item(),
                    'policy_tv': tv,
                    'val_diff': val_normal - val_zero,
                })

            mask = env.legal_actions()
            state, _, _, _ = env.step(np.random.choice(np.where(mask)[0]))

    return steps


def analyze_gate_activations(agent, num_players=3, seed=42):
    """Record per-step gate activations during a real game."""
    env = EcoEnv(num_players=num_players, seed=seed)
    state = env.reset()
    h = torch.zeros(1, 1, agent.lstm_hidden)
    c = torch.zeros(1, 1, agent.lstm_hidden)
    d = agent.lstm_hidden

    wi = agent.lstm.weight_ih_l0
    wh = agent.lstm.weight_hh_l0
    bih = agent.lstm.bias_ih_l0
    bhh = agent.lstm.bias_hh_l0

    steps = []
    with torch.no_grad():
        while not state.done:
            seat = int(state.current_player)
            if seat == 0:
                obs_t = make_obs(env, seat, num_players)
                shared = agent._shared_encode(obs_t)

                x = shared.squeeze(0)
                h_flat = h.squeeze(0).squeeze(0)
                gates = wi @ x + bih + wh @ h_flat + bhh

                i_gate = torch.sigmoid(gates[:d])
                f_gate = torch.sigmoid(gates[d:2*d])
                g_cell = torch.tanh(gates[2*d:3*d])
                o_gate = torch.sigmoid(gates[3*d:])

                new_info = i_gate * g_cell

                steps.append({
                    'i_mean': i_gate.mean().item(),
                    'f_mean': f_gate.mean().item(),
                    'g_norm': g_cell.norm().item(),
                    'o_mean': o_gate.mean().item(),
                    'new_info_norm': new_info.norm().item(),
                    'retained_norm': (f_gate * c.squeeze()).norm().item(),
                    'c_norm': c.norm().item(),
                    'h_norm': h.norm().item(),
                })

                _, (h, c) = agent.lstm(shared.unsqueeze(0), (h, c))

            mask = env.legal_actions()
            state, _, _, _ = env.step(np.random.choice(np.where(mask)[0]))

    return steps


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze LSTM recurrence usage")
    parser.add_argument("--checkpoint", default="model/ablation_ff_lstm/eco_latest.pkt")
    parser.add_argument("--num-games", type=int, default=500)
    parser.add_argument("--num-players", type=int, default=3)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--skip-games", action="store_true",
                        help="Skip win-rate games, only run analysis")
    args = parser.parse_args()

    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    agent = EcoAgentFFLSTM(num_players=args.num_players, lstm_hidden=args.lstm_hidden).cpu()
    agent.load_state_dict(sd)
    agent.eval()

    N = args.num_games
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Players: {args.num_players}, LSTM hidden: {args.lstm_hidden}")

    # ── 1. Gate activations ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. Gate activations during a real game")
    print("=" * 70)
    gate_steps = analyze_gate_activations(agent, args.num_players)
    print(f"{'Step':>4} | {'i_gate':>7} | {'f_gate':>7} | {'g_norm':>7} | {'o_gate':>7} | "
          f"{'new_info':>8} | {'retained':>8} | {'c_norm':>7}")
    print("-" * 80)
    for t, s in enumerate(gate_steps):
        if t < 5 or t >= len(gate_steps) - 3 or t % 10 == 0:
            print(f"{t:>4} | {s['i_mean']:>7.4f} | {s['f_mean']:>7.4f} | {s['g_norm']:>7.2f} | "
                  f"{s['o_mean']:>7.4f} | {s['new_info_norm']:>8.4f} | "
                  f"{s['retained_norm']:>8.4f} | {s['c_norm']:>7.2f}")

    # ── 2. History dependence ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. LSTM output: with history vs fresh state (same input)")
    print("=" * 70)
    hist_steps = analyze_history_dependence(agent, args.num_players)
    print(f"{'Step':>4} | {'cos_sim':>8} | {'diff/h':>8} | {'c_norm':>7} | {'h_norm':>7}")
    print("-" * 50)
    for t, s in enumerate(hist_steps):
        if t < 5 or t >= len(hist_steps) - 3 or t % 10 == 0:
            print(f"{t:>4} | {s['cos_sim']:>8.4f} | {s['diff_ratio']:>7.1%} | "
                  f"{s['c_norm']:>7.2f} | {s['h_norm']:>7.4f}")
    cos_vals = [s['cos_sim'] for s in hist_steps]
    diff_vals = [s['diff_ratio'] for s in hist_steps]
    print(f"\nMean cosine similarity: {np.mean(cos_vals):.4f}")
    print(f"Mean |diff|/|h|: {np.mean(diff_vals):.1%}")

    # ── 3. Trunk sensitivity ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. Trunk sensitivity: policy with vs without LSTM output")
    print("=" * 70)
    trunk_steps = analyze_trunk_sensitivity(agent, args.num_players)
    same = sum(s['action_same'] for s in trunk_steps)
    tv_vals = [s['policy_tv'] for s in trunk_steps]
    vd_vals = [s['val_diff'] for s in trunk_steps]
    print(f"Actions unchanged:    {same}/{len(trunk_steps)} ({same/len(trunk_steps):.0%})")
    print(f"Mean policy TV dist:  {np.mean(tv_vals):.4f}")
    print(f"Mean value diff:      {np.mean(vd_vals):+.4f}")

    # ── 4. Win rate ablations ─────────────────────────────────────────────
    if not args.skip_games:
        print("\n" + "=" * 70)
        print(f"4. Win rate ablations ({N} games vs random)")
        print("=" * 70)

        # Normal
        wr_normal = play_vs_random(agent, N, args.num_players)
        print(f"  Normal LSTM:          {wr_normal:.1%}")

        # Zero weight_hh
        agent_no_wh = EcoAgentFFLSTM(num_players=args.num_players,
                                     lstm_hidden=args.lstm_hidden).cpu()
        agent_no_wh.load_state_dict(sd)
        with torch.no_grad():
            agent_no_wh.lstm.weight_hh_l0.zero_()
            agent_no_wh.lstm.bias_hh_l0.zero_()
        agent_no_wh.eval()
        wr_no_wh = play_vs_random(agent_no_wh, N, args.num_players)
        print(f"  Zero weight_hh:       {wr_no_wh:.1%}")

        # Reset state every step
        wr_reset = play_vs_random(agent, N, args.num_players, reset_state=True)
        print(f"  Reset state/step:     {wr_reset:.1%}")

        print(f"\n  Normal - zero_wh:     {wr_normal - wr_no_wh:+.1%}")
        print(f"  Normal - reset:       {wr_normal - wr_reset:+.1%}")


if __name__ == "__main__":
    main()
