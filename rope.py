import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)  # (T, dim/2)

        # store as (T, dim/2)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())
        self.register_buffer("freqs", freqs)  # for visualization

    def forward(self, x):
        # x: (B, H, T, D) where D == head_dim == self.dim
        B, H, T, D = x.shape
        assert D == self.dim, f"RoPE dim mismatch: got {D}, expected {self.dim}"
        assert D % 2 == 0, "RoPE requires even head_dim"

        cos = self.cos[:T].unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1,1,T,D/2,1)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1,1,T,D/2,1)

        x1 = x[..., ::2]  # (B,H,T,D/2)
        x2 = x[..., 1::2] # (B,H,T,D/2)

        x_pair = torch.stack([x1, x2], dim=-1)  # (B,H,T,D/2,2)

        # rotation:
        # [x1'; x2'] = [ cos -sin; sin cos ] [x1; x2]
        x1p = x_pair[..., 0] * cos.squeeze(-1) - x_pair[..., 1] * sin.squeeze(-1)
        x2p = x_pair[..., 0] * sin.squeeze(-1) + x_pair[..., 1] * cos.squeeze(-1)

        out = torch.stack([x1p, x2p], dim=-1).flatten(-2)  # back to (B,H,T,D)
        return out

    @torch.no_grad()
    def get_freqs(self, T=None):
        T = self.max_seq_len if T is None else T
        return self.freqs[:T].detach().cpu()  # (T, D/2)