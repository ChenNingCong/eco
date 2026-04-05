#!/usr/bin/env python
"""Pull wandb metrics for active/recent runs and print smoothed summaries.

Usage:
    python scripts/check_wandb.py                  # all ablation/lstm runs
    python scripts/check_wandb.py ablation_baseline # filter by name
    python scripts/check_wandb.py --all             # show all runs
"""
import sys
import numpy as np
import wandb

METRICS = [
    "benchmark/vs_random/win_rate",
    "benchmark/vs_random/mean_score",
    "charts/episodic_return",
    "losses/entropy",
    "losses/explained_variance",
    "losses/approx_kl",
    "losses/policy_loss",
    "losses/value_loss",
    "grad/total_pre_clip",
    "grad/total_post_clip",
    "grad/lstm",
    "grad/flat_enc",
    "grad/actor_head",
    "grad/critic_head",
    "lstm/h_norm",
    "lstm/c_norm",
]


def ema_last(values, alpha=0.1):
    s = values[0]
    for v in values[1:]:
        s = alpha * v + (1 - alpha) * s
    return s


def report_run(run):
    print(f"\n{'='*70}")
    print(f"  {run.name}  (state={run.state})")
    print(f"{'='*70}")

    # Fetch last page of data only (page_size=500, no min_step = just get what's there)
    rows = []
    for row in run.scan_history(keys=["global_step"] + METRICS, page_size=500):
        rows.append(row)
        if len(rows) >= 500:
            break

    if not rows:
        print("  (no data yet)")
        return

    steps = [r["global_step"] for r in rows if "global_step" in r]
    print(f"  steps: {min(steps):.0f} → {max(steps):.0f}  ({len(rows)} points)")
    print()

    for m in METRICS:
        vals = [r[m] for r in rows if m in r and r[m] is not None]
        if not vals:
            continue
        smoothed = ema_last(vals)
        n_tail = min(50, len(vals))
        tail = vals[-n_tail:]
        print(f"  {m:<35}  ema={smoothed:>+10.4f}  "
              f"tail_{n_tail}: mean={np.mean(tail):>+10.4f}  std={np.std(tail):>8.4f}")


def main():
    api = wandb.Api(timeout=60)
    name_filter = sys.argv[1] if len(sys.argv) > 1 else None
    show_all = name_filter == "--all"

    if show_all:
        runs = list(api.runs("ppo-eco", order="-created_at"))[:20]
    else:
        runs = list(api.runs("ppo-eco", filters={
            "display_name": {"$regex": "ablation|lstm"}
        }, order="-created_at"))[:10]

    if name_filter and not show_all:
        runs = [r for r in runs if name_filter in r.name]

    if not runs:
        print("No matching runs found.")
        return

    for run in runs:
        report_run(run)


if __name__ == "__main__":
    main()
