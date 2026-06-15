import sys, torch, numpy as np
from game.lc.engine import LCEngine
from game.lc.agent import LCAgent
from game.lc.heuristic_player import HeuristicPlayer
from abstract.ppo_lstm import LSTMBatchedPlayer, LSTMSlicedPlayer
from abstract.player import RandomPlayer

dev = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = sys.argv[1] if len(sys.argv) > 1 else "model/lc_ppo_ffdiff_lr25e4/latest.pkt"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
agent = LCAgent(no_lstm=True, hidden_dim=512).to(dev)
agent.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)); agent.eval()
batched = LSTMBatchedPlayer(agent, dev, num_envs=1); ppo = LSTMSlicedPlayer(batched, 0)
print(f"loaded {ckpt} on {dev}", flush=True)

def play(eng, pls):
    eng.reset(); batched.reset()
    while not eng.done:
        p = eng.current_player
        eng.step(pls[p].action(eng.encode(p), eng.legal_actions().copy()))

def bench(name, p0, p1):
    eng = LCEngine(rng=np.random.default_rng(999), new_color_penalty=20)
    s0=s1=w0=lan=inv=pl=0.0
    for g in range(n):
        play(eng, {0:p0, 1:p1}); m0,m1 = eng.game_metrics(0), eng.game_metrics(1)
        s0+=m0["total_points"]; s1+=m1["total_points"]; w0+= m0["total_points"]>m1["total_points"]
        lan+=m0["open_lanes"]; inv+=m0["num_investment"]; pl+=m0["num_played"]
    print(f"{name:26s} s0={s0/n:6.1f} s1={s1/n:6.1f} s0win={w0/n*100:3.0f}%  "
          f"(s0 lanes={lan/n:.1f} inv={inv/n:.2f} played={pl/n:.1f})", flush=True)

H = HeuristicPlayer()
bench("PPO(0) vs Heuristic(1)", ppo, H)
bench("Heuristic(0) vs PPO(1)", H, ppo)
bench("PPO(0) vs Random(1)", ppo, RandomPlayer())
print("DONE", flush=True)
