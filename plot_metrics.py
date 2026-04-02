"""
Fetch wandb metrics and generate plots for documentation.

Usage: python plot_metrics.py
"""

import os
import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

OUT_DIR = "docs"
os.makedirs(OUT_DIR, exist_ok=True)

api = wandb.Api()
runs = api.runs("ningcong-chen/ppo-eco")

print("Available runs:")
for r in runs:
    print(f"  {r.name}  state={r.state}  id={r.id}")

# Find the best run (ent_gae_50m)
target_runs = {}
for r in runs:
    name = r.name
    for key in ["ent_gae_50m", "ent_gae_10m", "ent_gae", "ent_anneal", "gae95",
                "baseline", "no_shaping", "score_shortcut", "absolute_seat", "relative_seat"]:
        if key in name:
            if key not in target_runs or r.created_at > target_runs[key].created_at:
                target_runs[key] = r

print(f"\nFound runs: {list(target_runs.keys())}")


def fetch_history(run, keys, x_key="global_step"):
    """Fetch metrics from a wandb run, handling sparse logging."""
    hist = run.scan_history(keys=[x_key] + keys)
    data = {k: [] for k in [x_key] + keys}
    for row in hist:
        for k in [x_key] + keys:
            if k in row and row[k] is not None:
                data[k].append(row[k])
            else:
                data[k].append(None)
    # Filter rows where x_key is valid
    filtered = {k: [] for k in keys}
    filtered[x_key] = []
    for i in range(len(data[x_key])):
        if data[x_key][i] is not None:
            filtered[x_key].append(data[x_key][i])
            for k in keys:
                filtered[k].append(data[k][i])
    return filtered


def smooth(y, window=10):
    """Simple moving average, handling None values."""
    y_clean = [v if v is not None else np.nan for v in y]
    y_arr = np.array(y_clean, dtype=float)
    if len(y_arr) < window:
        return y_arr
    kernel = np.ones(window) / window
    # Pad to handle edges
    padded = np.concatenate([np.full(window // 2, y_arr[0]), y_arr, np.full(window // 2, y_arr[-1])])
    smoothed = np.convolve(padded, kernel, mode="same")[window // 2 : window // 2 + len(y_arr)]
    return smoothed


def plot_metric(run, metric_key, title, ylabel, filename, smooth_window=10, x_label="Training Steps"):
    data = fetch_history(run, [metric_key])
    x = np.array(data["global_step"])
    y = np.array([v if v is not None else np.nan for v in data[metric_key]])

    # Remove NaN
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        print(f"  No data for {metric_key}, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, alpha=0.25, color="steelblue", linewidth=0.5)
    if len(y) > smooth_window:
        ax.plot(x, smooth(y, smooth_window), color="steelblue", linewidth=2, label=f"Smoothed (w={smooth_window})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_comparison(runs_dict, metric_key, title, ylabel, filename, smooth_window=10):
    """Plot same metric across multiple runs for comparison."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_dict)))

    for (label, run), color in zip(runs_dict.items(), colors):
        data = fetch_history(run, [metric_key])
        x = np.array(data["global_step"])
        y = np.array([v if v is not None else np.nan for v in data[metric_key]])
        mask = ~np.isnan(y)
        x, y = x[mask], y[mask]
        if len(x) == 0:
            continue
        if len(y) > smooth_window:
            ax.plot(x, smooth(y, smooth_window), linewidth=2, label=label, color=color)
        else:
            ax.plot(x, y, linewidth=2, label=label, color=color)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot main metrics for the best run (ent_gae_50m) ──
best_key = "ent_gae_50m" if "ent_gae_50m" in target_runs else "ent_gae_10m"
if best_key in target_runs:
    best = target_runs[best_key]
    print(f"\nPlotting main metrics for: {best.name}")

    plot_metric(best, "benchmark/vs_random/win_rate",
                "Win Rate vs Random Opponent", "Win Rate (%)",
                "win_rate_vs_random.png", smooth_window=5)

    plot_metric(best, "losses/explained_variance",
                "Explained Variance", "Explained Variance",
                "explained_variance.png", smooth_window=20)

    plot_metric(best, "losses/entropy",
                "Policy Entropy", "Entropy",
                "entropy.png", smooth_window=20)

    plot_metric(best, "losses/value_loss",
                "Value Loss", "Value Loss",
                "value_loss.png", smooth_window=20)

    plot_metric(best, "losses/policy_loss",
                "Policy Loss", "Policy Loss",
                "policy_loss.png", smooth_window=20)

# ── Comparison plots across ablations ──
# Win rate comparison
wr_runs = {}
for key in ["ent_gae_50m", "ent_gae_10m", "ent_gae", "ent_anneal", "gae95"]:
    if key in target_runs:
        wr_runs[key] = target_runs[key]
if len(wr_runs) > 1:
    print(f"\nPlotting comparisons across: {list(wr_runs.keys())}")
    plot_comparison(wr_runs, "benchmark/vs_random/win_rate",
                    "Win Rate vs Random: Ablation Comparison", "Win Rate (%)",
                    "win_rate_comparison.png", smooth_window=3)
    plot_comparison(wr_runs, "losses/explained_variance",
                    "Explained Variance: Ablation Comparison", "Explained Variance",
                    "explained_variance_comparison.png", smooth_window=10)
    plot_comparison(wr_runs, "losses/entropy",
                    "Entropy: Ablation Comparison", "Entropy",
                    "entropy_comparison.png", smooth_window=10)

print("\nDone.")
