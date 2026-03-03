import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, hidden_mult=2.666, dropout=0.0, bias=False):
        super().__init__()

        hidden_dim = int(d_model * hidden_mult)

        # Two projections
        self.w1 = nn.Linear(d_model, hidden_dim, bias=bias)  # gate
        self.w3 = nn.Linear(d_model, hidden_dim, bias=bias)  # value

        # Output projection
        self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)

        gate = F.silu(self.w1(x))   # SiLU activation
        value = self.w3(x)

        out = gate * value          # elementwise multiply

        out = self.w2(out)
        out = self.dropout(out)

        return out