"""
sLSTM (Beck et al. 2024, xLSTM) — drop-in replacement for nn.LSTM.

Each hidden unit is independent in the recurrence (scalar weights r, not matrices).
Cross-unit mixing happens only through the input projection and surrounding layers.

Interface mirrors nn.LSTM:
    output, (h_n, extra_n) = slstm(input, (h_0, extra_0))

where `extra` packs (c, n, m) into one tensor of shape (num_layers, batch, 3*hidden).
This allows reusing LSTMState(h, c) everywhere — just c is wider.

Usage:
    slstm = SLSTM(input_size=256, hidden_size=128)
    h0 = torch.zeros(1, batch, 128)
    extra0 = torch.zeros(1, batch, 3 * 128)  # packed (c, n, m)
    output, (h_n, extra_n) = slstm(input_seq, (h0, extra0))
"""

import torch
import torch.nn as nn
import math


class SLSTMCell(nn.Module):
    """
    Single sLSTM cell. Independent per-unit recurrence with exponential gating.

    Args:
        input_size: dimension of input x_t
        hidden_size: number of independent recurrent units
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input projections: w^T @ x_t → (4 * hidden_size)
        # Order: [z (cell input), i (input gate), f (forget gate), o (output gate)]
        self.W = nn.Linear(input_size, 4 * hidden_size)

        # Recurrent scalar weights: r * h_{t-1} → (4 * hidden_size)
        # One scalar per unit per gate — NOT a matrix
        self.r = nn.Parameter(torch.empty(4 * hidden_size))

        self._init_weights()

    def _init_weights(self):
        d = self.hidden_size
        # Gate weights: small_init_init_ (Nguyen & Salazar 2019, used by xLSTM)
        std = math.sqrt(2.0 / (5.0 * self.input_size))
        nn.init.normal_(self.W.weight, std=std)
        nn.init.zeros_(self.W.bias)

        # Forget gate bias: large positive (xLSTM uses range [3, 6])
        # We use uniform [3, 6] per unit — exp(3)≈20, exp(6)≈403
        self.W.bias.data[2 * d : 3 * d].uniform_(3.0, 6.0)

        # Recurrent scalars: zero (xLSTM default — start without recurrence)
        nn.init.zeros_(self.r)

    def forward(self, x_t: torch.Tensor, c: torch.Tensor, n: torch.Tensor,
                h: torch.Tensor, m: torch.Tensor):
        """
        Single timestep.

        Args:
            x_t: (batch, input_size)
            c: (batch, hidden_size) — cell state (numerator)
            n: (batch, hidden_size) — normalizer state (denominator)
            h: (batch, hidden_size) — previous hidden output
            m: (batch, hidden_size) — log-space stabilizer

        Returns:
            h_new: (batch, hidden_size)
            c_new, n_new, h_new, m_new: updated states
        """
        d = self.hidden_size

        # Pre-activations: W @ x + r * h + b
        # r is (4d,), h is (B, d) → expand h to (B, 4d) for elementwise multiply
        pre = self.W(x_t) + self.r * h.repeat(1, 4)   # (B, 4d)
        z_pre, i_pre, f_pre, o_pre = pre.split(d, dim=-1)

        # Gate activations
        z_t = torch.tanh(z_pre)          # cell input ∈ (-1, 1)
        o_t = torch.sigmoid(o_pre)       # output gate ∈ (0, 1)

        # Log-space stabilization for exponential gates
        m_new = torch.max(f_pre + m, i_pre)
        f_t = torch.exp(f_pre + m - m_new)    # stabilized forget ∈ [0, ~1]
        i_t = torch.exp(i_pre - m_new)        # stabilized input ∈ [0, ~1]

        # State update
        c_new = f_t * c + i_t * z_t      # numerator
        n_new = f_t * n + i_t            # denominator

        # Output: bounded weighted average
        h_tilde = c_new / (n_new.abs() + 1e-6)
        h_new = o_t * h_tilde

        return h_new, c_new, n_new, m_new


class SLSTM(nn.Module):
    """
    sLSTM module — drop-in replacement for nn.LSTM.

    State is packed to match nn.LSTM's (h, c) interface:
      - h: (num_layers, batch, hidden_size) — hidden state (same as LSTM)
      - extra: (num_layers, batch, 3 * hidden_size) — packed (c, n, m)

    This means LSTMState(h, c) works if c is allocated with 3 * hidden_size.

    Args:
        input_size: input feature dimension
        hidden_size: number of recurrent units per layer
        num_layers: number of stacked sLSTM layers
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            inp = input_size if layer == 0 else hidden_size
            self.cells.append(SLSTMCell(inp, hidden_size))

    def forward(self, input: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        """
        Args:
            input: (seq_len, batch, input_size)
            state: (h, extra) where
                h: (num_layers, batch, hidden_size)
                extra: (num_layers, batch, 3 * hidden_size)

        Returns:
            output: (seq_len, batch, hidden_size)
            (h_n, extra_n): final state, same shapes as input state
        """
        h_prev, extra = state
        d = self.hidden_size
        L = self.num_layers

        # Unpack per-layer states (no in-place modification — use lists)
        h_list = [h_prev[l] for l in range(L)]
        c_list = [extra[l, :, 0*d : 1*d] for l in range(L)]
        n_list = [extra[l, :, 1*d : 2*d] for l in range(L)]
        m_list = [extra[l, :, 2*d : 3*d] for l in range(L)]

        seq_len = input.shape[0]
        outputs = []

        for t in range(seq_len):
            x = input[t]  # (batch, input_size)
            for layer, cell in enumerate(self.cells):
                h_new, c_new, n_new, m_new = cell(
                    x, c_list[layer], n_list[layer], h_list[layer], m_list[layer]
                )
                h_list[layer] = h_new
                c_list[layer] = c_new
                n_list[layer] = n_new
                m_list[layer] = m_new
                x = h_new  # input to next layer
            outputs.append(x)

        output = torch.stack(outputs, dim=0)

        # Pack states back (new tensors, no in-place)
        h_out = torch.stack(h_list, dim=0)
        extra_out = torch.cat([
            torch.stack(c_list, dim=0),
            torch.stack(n_list, dim=0),
            torch.stack(m_list, dim=0),
        ], dim=-1)
        return output, (h_out, extra_out)


class SLSTMBlock(nn.Module):
    """
    Pre-norm residual sLSTM block (xLSTM-style).

        out = x + out_proj(slstm(pre_norm(x)))

    When input_size == hidden_size, this is a clean residual block.
    When they differ, out_proj maps back to input_size.

    State format same as SLSTM: (h, extra) where extra packs (c, n, m).

    Args:
        input_size: dimension of input/output (residual dimension)
        hidden_size: sLSTM recurrent dimension
        num_layers: stacked sLSTM layers inside the block
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.pre_norm = nn.RMSNorm(input_size)
        self.slstm = SLSTM(input_size, hidden_size, num_layers)
        self.out_proj = nn.Linear(hidden_size, input_size)
        # Small init: residual starts near identity, but sLSTM gets gradient from step 1
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        """
        Args:
            input: (seq_len, batch, input_size)
            state: (h, extra) for the inner SLSTM

        Returns:
            output: (seq_len, batch, input_size) — residual output
            (h_n, extra_n): updated state
        """
        # Pre-norm: normalize along feature dim
        normed = self.pre_norm(input)
        # sLSTM recurrence
        slstm_out, new_state = self.slstm(normed, state)
        # Project back + residual
        output = input + self.out_proj(slstm_out)
        return output, new_state


def make_slstm_state(num_layers: int, batch_size: int, hidden_size: int,
                     device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create zero-initialized sLSTM state matching SLSTM's (h, extra) format."""
    h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    extra = torch.zeros(num_layers, batch_size, 3 * hidden_size, device=device)
    # Initialize n=1 (normalizer) to avoid division by zero on first step
    extra[:, :, hidden_size : 2 * hidden_size] = 1.0
    return h, extra


# ── torch.compile wrapper ─────────────────────────────────────────────────

def compile_slstm(model: SLSTM) -> SLSTM:
    """Compile the sLSTM cells for faster execution."""
    for i, cell in enumerate(model.cells):
        model.cells[i] = torch.compile(cell)
    return model
