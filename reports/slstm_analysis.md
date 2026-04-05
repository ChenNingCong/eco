# sLSTM Ablation Analysis Report

**Run**: `ablation_ff_slstm` (wandb: `7pu404bf`)
**Date**: 2026-04-05
**Status**: Running (~78%, 7.8M/10M steps)
**Architecture**: FF 2-layer encoder + SLSTMBlock residual + 2-layer trunks (H=256, sLSTM hidden=128)

## Executive Summary

The sLSTM residual block **fails to learn meaningful recurrence**. Checkpoint analysis reveals that the recurrent scalar weights (`r`) remain near-zero throughout training, meaning the cell ignores its previous hidden state entirely. The block effectively degenerates into an identity residual + learned bias offset, contributing no temporal memory. Performance may eventually catch up to the FF baseline via the residual path, but the sLSTM itself is not contributing any temporal reasoning.

## 1. Training Metrics (from wandb)

| Step | h_norm | c_norm | Entropy |
|-----:|-------:|-------:|--------:|
| 3.0M | 0.020  | 774    | -       |
| 3.2M | 0.021  | 784    | -       |
| 4.4M | 0.250  | 848    | -       |
| 5.1M | 0.286  | 895    | -       |
| 5.3M | 0.117  | 836    | -       |
| 6.5M | 0.583  | 793    | -       |
| 7.3M | 0.168  | 786    | -       |
| 7.5M | 0.134  | 701    | -       |

**Summary at 7.8M**: win_rate=92.5%, h_norm=0.36, c_norm=814

Two anomalies were reported:
1. **c_norm ~750**: appears alarmingly large
2. **h_norm blows up 10-20x**: grew from ~0.02 to ~0.58 between 3M and 6.5M steps

Both are explained by the analysis below.

## 2. Checkpoint Parameter Analysis

Sampled 10 checkpoints evenly from 82K to 6.6M steps.

### 2.1 Parameter Norm Evolution

| Step | r_norm | r_mean | W_norm | bias_norm | out_w_norm | out_b_norm |
|-----:|-------:|-------:|-------:|----------:|-----------:|-----------:|
|  82K | 0.003  | -0.000 | 14.32  | 51.83     | 2.03       | 0.04       |
| 737K | 0.037  |  0.000 | 14.41  | 51.84     | 2.37       | 0.74       |
| 1.5M | 0.036  |  0.000 | 14.42  | 51.85     | 2.53       | 1.45       |
| 2.2M | 0.036  |  0.000 | 14.43  | 51.85     | 2.59       | 1.92       |
| 2.9M | 0.034  |  0.000 | 14.43  | 51.85     | 2.65       | 2.33       |
| 3.7M | 0.034  |  0.000 | 14.44  | 51.86     | 2.67       | 2.39       |
| 4.4M | 0.041  |  0.000 | 14.46  | 51.86     | 2.75       | 2.61       |
| 5.2M | 0.045  |  0.000 | 14.47  | 51.86     | 2.84       | 2.78       |
| 5.9M | 0.054  |  0.000 | 14.55  | 51.85     | 3.09       | 2.90       |
| 6.6M | 0.062  |  0.000 | 14.58  | 51.85     | 3.15       | 2.96       |

**Key observations**:
- **`r` (recurrent weights) stay near zero**: r_norm grows from 0.003 to 0.062 over 6.6M steps. For context, with hidden_size=128 and 4 gates, there are 512 scalar weights — the per-element magnitude is ~0.003. The cell essentially has no recurrence.
- **Gate biases are frozen**: bias_norm barely moves (51.83 → 51.85). The large norm is dominated by the forget gate bias, initialized U[3,6].
- **`out_proj` bias is the main learnable signal**: grows from 0.04 to 2.96, meaning the block learns a constant additive shift in the residual stream, not a temporal representation.
- **`out_proj` weight also grows** (2.03 → 3.15), but slowly — the block is learning a weak linear projection, not leveraging recurrent state.

### 2.2 Per-Gate Breakdown (First vs Last Checkpoint)

**Step 82K (near initialization):**
| Gate | r_mean | r_std | bias_mean | bias_std |
|------|-------:|------:|----------:|---------:|
| z (cell input)  | -0.0000 | 0.0000 | 0.0001  | 0.0009 |
| i (input gate)  |  0.0000 | 0.0000 | -0.0000 | 0.0008 |
| f (forget gate)  | -0.0000 | 0.0000 | **4.502** | 0.851  |
| o (output gate)  | -0.0000 | 0.0003 | -0.0001 | 0.0005 |

**Step 6.6M (latest):**
| Gate | r_mean | r_std | bias_mean | bias_std |
|------|-------:|------:|----------:|---------:|
| z (cell input)  |  0.0002 | 0.0031 | -0.0005 | 0.0047 |
| i (input gate)  | -0.0000 | 0.0005 | -0.0018 | 0.0052 |
| f (forget gate)  |  0.0002 | 0.0012 | **4.504** | 0.850  |
| o (output gate)  |  0.0000 | 0.0043 | -0.0051 | 0.0079 |

The forget gate bias is essentially unchanged from initialization (4.50 ± 0.85). All other biases remain near zero. The recurrent weights `r` for all gates are negligible.

## 3. Explaining the Anomalies

### 3.1 Why c_norm ~ 750

The logged `c_norm` is computed on the packed `extra` tensor, which has shape `(layers, batch, 3 * hidden_size)` and contains three concatenated components:

```
extra = [c (cell state) | n (normalizer) | m (log-stabilizer)]
```

The `m` component is the log-space stabilizer, defined as:
```python
m_new = max(f_pre + m, i_pre)
```

With forget bias ≈ 4.5, this means `f_pre ≈ 4.5` at each step. Since `m` starts at 0 and `f_pre + m > i_pre` (because i_bias ≈ 0), `m` grows by ~4.5 per timestep. Over a typical game of ~100 steps, `m ≈ 450` per unit. The Frobenius norm across 128 units gives:

```
|m| ≈ 450 × sqrt(128) ≈ 5,091
```

Per-environment normalization brings this down, but it still dominates the packed tensor's norm. **The large c_norm is an artifact of including the monotonically-growing (but harmless) log-stabilizer `m`.**

Forward-pass simulation confirms: with fresh state, `m` grows ~52 per step (matching `f_pre ≈ 4.5` × sqrt(128) ≈ 51).

### 3.2 Why h_norm blows up

The h_norm growth (0.02 → 0.58 over training) is **not** from recurrent state dynamics — since `r ≈ 0`, there is no recurrent contribution. Instead, it comes from the `out_proj` bias growing:

```python
# SLSTMBlock.forward():
output = input + self.out_proj(slstm_out)  # residual
```

As `out_proj.bias` grows from 0.04 to 2.96, the block adds a larger constant offset to the residual stream. This offset flows through to `h`, which is the sLSTM hidden output. Since the out_proj was initialized near zero (std=0.01) as a "start near identity" strategy, any learning shows up as h_norm growth.

The growth is not a stability issue — it's the block learning the only thing it can: a fixed bias in the residual path.

## 4. Gate-by-Gate State Summary

The sLSTM cell has 4 gates and 4 state variables. Here is the observed state of each:

### Initialization

All parameters come from `SLSTMCell._init_weights()` and `SLSTMBlock.__init__()`:

| Parameter | Initialization | Notes |
|-----------|---------------|-------|
| `W.weight` (input projection) | Normal(0, std=sqrt(2/(5*input_size))) ≈ Normal(0, 0.018) | Shared 4d matrix for all gates |
| `W.bias[z]` (cell input bias) | **0** | |
| `W.bias[i]` (input gate bias) | **0** | |
| `W.bias[f]` (forget gate bias) | **Uniform[3, 6]** | xLSTM default; exp(3)≈20, exp(6)≈403 |
| `W.bias[o]` (output gate bias) | **0** | |
| `r` (recurrent scalars, all gates) | **0** | xLSTM default — start without recurrence |
| `out_proj.weight` | Normal(0, std=0.01) | Small init so residual starts near identity |
| `out_proj.bias` | **0** | |
| `pre_norm` (RMSNorm) | weight = **1** (PyTorch default) | |

Initial state (per episode reset):
| State | Initial value |
|-------|--------------|
| `c` | **0** |
| `n` | **1** (to avoid division by zero in `c/n`) |
| `m` | **0** |
| `h` | **0** |

### Gates (from checkpoint weights at 6.6M steps)

| Gate | Role | Activation | Init bias → init activation | Bias at 6.6M | r at 6.6M | Observed state |
|------|------|-----------|----------------------------|--------------|-----------|----------------|
| **z** (cell input) | New info to write | tanh | 0 → tanh(0) = 0 | ≈ 0 (mean=-0.0005) | ≈ 0 (std=0.003) | Functional but gated by `i_t` |
| **i** (input gate) | How much new info enters | exp (stabilized) | 0 → suppressed by f (see below) | ≈ 0 (mean=-0.002) | ≈ 0 (std=0.0005) | **Suppressed** — `i_t ≈ exp(-4.5) ≈ 0.01` at step 1, vanishes further |
| **f** (forget gate) | How much old state to keep | exp (stabilized) | **U[3,6]** → `f_t ≈ 1` (dominates max) | **≈ 4.5** (mean=4.504) | ≈ 0 (std=0.001) | **Saturated at ~1** — dominates the `max(...)` stabilizer |
| **o** (output gate) | How much cell to expose | sigmoid | 0 → sigmoid(0) = 0.5 | ≈ 0 (mean=-0.005) | ≈ 0 (std=0.004) | Half-open ≈ 0.5, but cell is ~empty so doesn't matter |

Key: biases barely moved from initialization across 6.6M steps of training.

### State variables

| State | Role | Init | Observed state |
|-------|------|------|----------------|
| **c** (cell / numerator) | Accumulated memory | 0 | **~0** — `i_t` is too small to write anything in, `f_t ≈ 1` preserves the nothing |
| **n** (normalizer / denominator) | Normalizes output `h = o * c/n` | 1 | Stable ≈ 1 per unit (`f_t*n + i_t ≈ 1*1 + 0.01`) |
| **m** (log stabilizer) | Numerical trick, not "real" state | 0 | Grows ~4.5 per step (harmless, cancels in exp) |
| **h** (hidden output) | Cell output → next layer | 0 | **~0** — `c ≈ 0` and `o ≈ 0.5`, so `h = 0.5 * 0/1 ≈ 0` |

### How the gates interact

```
m_new = max(f_pre + m, i_pre)         # forget side always wins (4.5 + m >> 0)
f_t   = exp(f_pre + m - m_new) ≈ 1    # retains everything
i_t   = exp(i_pre - m_new)    ≈ 0     # blocked by large m_new

c_new = 1 * c + 0 * z_t ≈ c ≈ 0      # nothing in, nothing changes
n_new = 1 * n + 0        ≈ n ≈ 1      # stable
h_new = sigmoid(0) * (0 / 1) ≈ 0      # zero output
```

The recurrent scalar weights `r` are all ≈ 0, meaning `h_{t-1}` has no influence on any gate. All gate pre-activations come solely from the current input `W @ x_t + bias`.

### Net effect on the residual block

```python
# SLSTMBlock.forward():
slstm_out ≈ 0               # sLSTM produces near-zero output
output = input + out_proj(0) # ≈ input + out_proj.bias
```

The block degenerates into `identity + learned constant bias`. The sLSTM contributes no temporal information.

## 5. Why the sLSTM Didn't Learn Recurrence

The sLSTM block is in a residual configuration (`output = input + out_proj(slstm_out)`), so a dead sLSTM simply means the model falls back to the FF path. Performance should eventually converge to FF-level — the current 92.5% at 78% training likely reflects slower optimization rather than a permanent deficit. The `ablation_ff_seqmb` run (FF + sequential MB, no sLSTM) will confirm whether the slower convergence is from sequential minibatching or the dead sLSTM block adding noise.

The core question is: **why did the sLSTM fail to learn any recurrence?**

The forget gate bias (initialized [3, 6], mean ≈ 4.5) creates an asymmetry in the `max(f_pre + m, i_pre)` stabilizer: the forget side always wins, suppressing `i_t` exponentially. With no input entering the cell, there's no gradient signal to learn recurrence through `r`. The optimizer instead uses the residual path, which works immediately without needing to learn.

In standard LSTM, `sigmoid(f)` and `sigmoid(i)` are independent — a large forget bias doesn't suppress the input gate. In sLSTM, the shared log-space stabilizer couples them in a winner-take-all competition.

## 6. Full Comparison Table

| Run | Architecture | MB Strategy | Win Rate | Status |
|-----|-------------|-------------|----------|--------|
| ablation_ff | FF 2-layer | random | 98.5% | finished |
| ablation_ff_lstm | FF 2-layer + LSTM concat | sequential | 98.0% | finished |
| ablation_cell_clip | LSTM 1-layer + clip | sequential | 94.0% | finished |
| ablation_baseline | LSTM 1-layer | sequential | 92.0% | finished |
| **ablation_ff_slstm** | **FF 2-layer + sLSTM block** | **sequential** | **92.5%** | **running (78%)** |
| ablation_ff_seqmb | FF 2-layer | sequential | ? | running (early) |

## 6. Methodology

Analysis performed by loading `.pkt` checkpoints directly from `model/ablation_ff_slstm/`:
- 10 checkpoints sampled evenly across 82K–6.6M steps
- Per-gate decomposition of recurrent weights `r` and biases
- Forward-pass simulation through the sLSTM cell with trained weights
- wandb metrics corroborated checkpoint findings (h_norm, c_norm trends)

Script: `analyze_slstm_state.py`
