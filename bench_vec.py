"""Vectorized heuristic<->PPO benchmark via VecSinglePlayerEnv (128 parallel envs).
Mirrors PPOLSTMTrainer._run_benchmark. Usage: python bench_vec.py <ckpt> [n_games]"""
import sys, torch, numpy as np
from abstract import VecSinglePlayerEnv, RandomPlayer, key_from_seed
from abstract.ppo_lstm import LSTMBatchedPlayer, obs_to_tensor, make_lstm_state
from game.lc.factory import LCEnvFactory
from game.lc.agent import LCAgent
from game.lc.heuristic_player import HeuristicPlayer

dev = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = sys.argv[1] if len(sys.argv) > 1 else "model/lc_ppo_ffdiff_lr25e4/latest.pkt"
N_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 800
N = 128
factory = LCEnvFactory(new_color_penalty=20)

agent = LCAgent(no_lstm=True, hidden_dim=512).to(dev)
agent.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)); agent.eval()
print(f"loaded {ckpt} on {dev}", flush=True)


def run(label, opponent, agent_is_ppo):
    envs = VecSinglePlayerEnv(num_envs=N, opponent=opponent, env_factory=factory, key=key_from_seed(777))
    obs, masks = envs.reset()
    h = HeuristicPlayer()
    if agent_is_ppo:
        lstm = make_lstm_state(agent.lstm_layers, N, agent.lstm_hidden, dev)
        done = torch.zeros(N, device=dev)
    tot = wins = 0
    sc, lan, inv, pl = [], [], [], []
    while tot < N_GAMES:
        if agent_is_ppo:
            with torch.no_grad():
                a, _, _, _, lstm = agent.get_action_and_value(
                    obs_to_tensor(obs, dev),
                    torch.as_tensor(masks, dtype=torch.bool, device=dev), lstm, done)
            a = a.cpu().numpy()
        else:
            a = h.batch_action(obs, masks, list(range(N)))
        obs, masks, rewards, term, trunc, infos = envs.step(a)
        if agent_is_ppo:
            done = torch.as_tensor(term, dtype=torch.float32, device=dev)
        for i, t in enumerate(term):
            if not t:
                continue
            tot += 1
            scores = infos[i].get("final_scores"); seat = infos[i].get("agent_seat", 0)
            if scores is not None:
                s = np.asarray(scores); sc.append(float(s[seat]))
                wins += int(s[seat] > s.min())
            gm = infos[i].get("game_metrics")
            if gm:
                lan.append(gm["open_lanes"]); inv.append(gm["num_investment"]); pl.append(gm["num_played"])
            if agent_is_ppo:
                lstm.h[:, i] = 0; lstm.c[:, i] = 0
    envs.close()
    print(f"{label:30s} score={np.mean(sc):6.1f}  win={wins/tot*100:3.0f}%  "
          f"lanes={np.mean(lan):.1f} inv={np.mean(inv):.2f} played={np.mean(pl):.1f}  (n={tot})", flush=True)


run("PPO  vs Heuristic", HeuristicPlayer(), agent_is_ppo=True)
run("Heuristic vs PPO", LSTMBatchedPlayer(agent, dev, N), agent_is_ppo=False)
run("PPO  vs Random", RandomPlayer(), agent_is_ppo=True)
run("Heuristic vs Random", RandomPlayer(), agent_is_ppo=False)
print("DONE", flush=True)
