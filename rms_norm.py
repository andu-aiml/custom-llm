import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape: (batch, seq_len, dim)

        # Compute RMS
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        # Normalize
        x_norm = x / rms

        return self.scale * x_norm