"""
Analyze sLSTM internal states (c, n, m, h) across training checkpoints.

Loads saved .pkt checkpoints from an sLSTM model and reports the parameter norms
for the sLSTM block — specifically the recurrent scalar weights (r), gate biases,
and out_proj weights. Also checks if there's any drift in the learned parameters.
"""
import torch
import os
import sys
import glob
import numpy as np

def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)

def analyze_slstm_params(state_dict):
    """Extract sLSTM-specific parameter stats from a state_dict."""
    stats = {}
    for key, val in state_dict.items():
        if "slstm" not in key:
            continue
        stats[key] = {
            "shape": tuple(val.shape),
            "norm": val.norm().item(),
            "mean": val.mean().item(),
            "std": val.std().item(),
            "min": val.min().item(),
            "max": val.max().item(),
        }
    return stats

def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "model/ablation_ff_slstm"

    # Find all checkpoints, sort by step
    files = glob.glob(os.path.join(model_dir, "eco_*.pkt"))
    ckpts = []
    for f in files:
        base = os.path.basename(f).replace("eco_", "").replace(".pkt", "")
        if base == "latest":
            continue
        try:
            step = int(base)
            ckpts.append((step, f))
        except ValueError:
            continue
    ckpts.sort()

    if not ckpts:
        print(f"No checkpoints found in {model_dir}")
        return

    # Sample ~10 checkpoints evenly across training
    if len(ckpts) > 10:
        indices = np.linspace(0, len(ckpts) - 1, 10, dtype=int)
        ckpts = [ckpts[i] for i in indices]

    print(f"Analyzing {len(ckpts)} checkpoints from {model_dir}")
    print("=" * 100)

    # Focus on key sLSTM parameters
    key_params = [
        "slstm_block.slstm.cells.0.r",       # recurrent scalar weights
        "slstm_block.slstm.cells.0.W.weight", # input projection
        "slstm_block.slstm.cells.0.W.bias",   # gate biases (includes forget bias)
        "slstm_block.out_proj.weight",         # output projection
        "slstm_block.out_proj.bias",
        "slstm_block.pre_norm.weight",         # RMSNorm
    ]

    print(f"\n{'Step':>10s} | {'r_norm':>8s} | {'r_mean':>8s} | {'W_norm':>8s} | {'bias_norm':>8s} | {'out_w_norm':>8s} | {'out_b_norm':>8s} | {'norm_w':>8s}")
    print("-" * 100)

    for step, path in ckpts:
        sd = load_checkpoint(path)

        vals = {}
        for key in key_params:
            for sd_key in sd:
                if sd_key.endswith(key):
                    vals[key] = sd[sd_key]
                    break

        r = vals.get("slstm_block.slstm.cells.0.r")
        W_w = vals.get("slstm_block.slstm.cells.0.W.weight")
        W_b = vals.get("slstm_block.slstm.cells.0.W.bias")
        out_w = vals.get("slstm_block.out_proj.weight")
        out_b = vals.get("slstm_block.out_proj.bias")
        norm_w = vals.get("slstm_block.pre_norm.weight")

        print(f"{step:>10d} | {r.norm().item():>8.4f} | {r.mean().item():>8.4f} | {W_w.norm().item():>8.4f} | {W_b.norm().item():>8.4f} | {out_w.norm().item():>8.4f} | {out_b.norm().item():>8.4f} | {norm_w.norm().item() if norm_w is not None else 0:>8.4f}")

    # Detailed analysis of the recurrent weights and gate biases at first and last checkpoint
    for label, (step, path) in [("FIRST", ckpts[0]), ("LAST", ckpts[-1])]:
        sd = load_checkpoint(path)

        r = None
        W_b = None
        for sd_key in sd:
            if sd_key.endswith("slstm_block.slstm.cells.0.r"):
                r = sd[sd_key]
            if sd_key.endswith("slstm_block.slstm.cells.0.W.bias"):
                W_b = sd[sd_key]

        if r is None:
            continue

        d = r.shape[0] // 4  # hidden_size

        print(f"\n{'='*60}")
        print(f"{label} checkpoint (step {step})")
        print(f"{'='*60}")

        # Recurrent weights per gate
        r_z, r_i, r_f, r_o = r[:d], r[d:2*d], r[2*d:3*d], r[3*d:]
        print(f"\nRecurrent scalar weights r (hidden_size={d}):")
        print(f"  z (cell input): mean={r_z.mean():.4f}, std={r_z.std():.4f}, min={r_z.min():.4f}, max={r_z.max():.4f}")
        print(f"  i (input gate):  mean={r_i.mean():.4f}, std={r_i.std():.4f}, min={r_i.min():.4f}, max={r_i.max():.4f}")
        print(f"  f (forget gate): mean={r_f.mean():.4f}, std={r_f.std():.4f}, min={r_f.min():.4f}, max={r_f.max():.4f}")
        print(f"  o (output gate): mean={r_o.mean():.4f}, std={r_o.std():.4f}, min={r_o.min():.4f}, max={r_o.max():.4f}")

        # Gate biases
        b_z, b_i, b_f, b_o = W_b[:d], W_b[d:2*d], W_b[2*d:3*d], W_b[3*d:]
        print(f"\nGate biases:")
        print(f"  z (cell input): mean={b_z.mean():.4f}, std={b_z.std():.4f}")
        print(f"  i (input gate):  mean={b_i.mean():.4f}, std={b_i.std():.4f}")
        print(f"  f (forget gate): mean={b_f.mean():.4f}, std={b_f.std():.4f}")
        print(f"  o (output gate): mean={b_o.mean():.4f}, std={b_o.std():.4f}")

    # Now do a forward-pass analysis: load a checkpoint and run dummy data through
    # to see actual gate activations and state magnitudes
    print(f"\n{'='*60}")
    print("Forward-pass gate analysis (last checkpoint)")
    print(f"{'='*60}")

    step, path = ckpts[-1]
    sd = load_checkpoint(path)

    from slstm import SLSTMCell

    # Find hidden size from r parameter
    for sd_key in sd:
        if sd_key.endswith("slstm_block.slstm.cells.0.r"):
            d = sd[sd_key].shape[0] // 4
            break

    # Find input size from W.weight
    for sd_key in sd:
        if sd_key.endswith("slstm_block.slstm.cells.0.W.weight"):
            input_size = sd[sd_key].shape[1]
            break

    cell = SLSTMCell(input_size, d)
    # Load cell weights
    cell_sd = {}
    for sd_key in sd:
        if "slstm_block.slstm.cells.0." in sd_key:
            short_key = sd_key.split("slstm_block.slstm.cells.0.")[-1]
            cell_sd[short_key] = sd[sd_key]
    cell.load_state_dict(cell_sd)
    cell.eval()

    # Simulate with random input for a few steps to see gate behavior
    B = 1
    x = torch.randn(B, input_size) * 0.1  # small input
    c = torch.zeros(B, d)
    n = torch.ones(B, d)  # normalizer starts at 1
    h = torch.zeros(B, d)
    m = torch.zeros(B, d)

    print(f"\nSimulating {20} steps with random input (scale=0.1):")
    print(f"{'Step':>4s} | {'|h|':>8s} | {'|c|':>8s} | {'|n|':>8s} | {'|m|':>8s} | {'c/n':>8s}")
    print("-" * 60)

    with torch.no_grad():
        for t in range(20):
            # Peek at gate values before step
            pre = cell.W(x) + cell.r * h.repeat(1, 4)
            z_pre, i_pre, f_pre, o_pre = pre.split(d, dim=-1)

            h, c, n, m = cell(x, c, n, h, m)
            c_over_n = (c / (n.abs() + 1e-6)).norm().item()

            if t < 5 or t >= 15:
                print(f"{t:>4d} | {h.norm().item():>8.4f} | {c.norm().item():>8.4f} | {n.norm().item():>8.4f} | {m.norm().item():>8.4f} | {c_over_n:>8.4f}")

                if t == 0:
                    # Print gate activations for first step
                    m_new = torch.max(f_pre + m, i_pre)  # but m was 0 before this
                    # We can't easily reconstruct mid-step, but biases tell the story
                    print(f"      f_pre: mean={f_pre.mean():.3f}, i_pre: mean={i_pre.mean():.3f}, o_pre: mean={o_pre.mean():.3f}")
            elif t == 5:
                print(f"  ...")

    # Also check: what do forget gate biases look like over training?
    print(f"\n{'='*60}")
    print("Forget gate bias evolution across training")
    print(f"{'='*60}")
    print(f"{'Step':>10s} | {'f_bias_mean':>11s} | {'f_bias_min':>10s} | {'f_bias_max':>10s} | {'i_bias_mean':>11s}")
    print("-" * 70)

    for step, path in ckpts:
        sd = load_checkpoint(path)
        for sd_key in sd:
            if sd_key.endswith("slstm_block.slstm.cells.0.W.bias"):
                W_b = sd[sd_key]
                d = W_b.shape[0] // 4
                b_i, b_f = W_b[d:2*d], W_b[2*d:3*d]
                print(f"{step:>10d} | {b_f.mean():>11.4f} | {b_f.min():>10.4f} | {b_f.max():>10.4f} | {b_i.mean():>11.4f}")
                break


if __name__ == "__main__":
    main()
