import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQAAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        rope,                 # instance of your RoPE(head_dim, max_seq_len)
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads

        self.rope = rope
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Projections:
        # Q: (d_model -> n_heads * head_dim)  == d_model
        # K,V: (d_model -> n_kv_heads * head_dim)  smaller than d_model if n_kv_heads < n_heads
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)

        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)

    def _shape_q(self, q, B, T):
        # (B, T, n_heads*head_dim) -> (B, n_heads, T, head_dim)
        return q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, kv, B, T, n_kv):
        # (B, T, n_kv_heads*head_dim) -> (B, n_kv_heads, T, head_dim)
        return kv.view(B, T, n_kv, self.head_dim).transpose(1, 2)

    def _repeat_kv(self, kv):
        """
        kv: (B, n_kv_heads, T, head_dim)
        return: (B, n_heads, T, head_dim) by repeating each kv head group_size times
        """
        # Repeat along head axis
        return kv.repeat_interleave(self.group_size, dim=1)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        B, T, _ = x.shape

        q = self.wq(x)  # (B, T, n_heads*head_dim)
        k = self.wk(x)  # (B, T, n_kv_heads*head_dim)
        v = self.wv(x)  # (B, T, n_kv_heads*head_dim)

        q = self._shape_q(q, B, T)                      # (B, n_heads, T, head_dim)
        k = self._shape_kv(k, B, T, self.n_kv_heads)    # (B, n_kv_heads, T, head_dim)
        v = self._shape_kv(v, B, T, self.n_kv_heads)    # (B, n_kv_heads, T, head_dim)

        # Apply RoPE to Q and K
        q = self.rope(q)                                # (B, n_heads, T, head_dim)
        k = self.rope(k)                                # (B, n_kv_heads, T, head_dim)

        # Expand K,V to match Q heads
        k = self._repeat_kv(k)                          # (B, n_heads, T, head_dim)
        v = self._repeat_kv(v)                          # (B, n_heads, T, head_dim)

        # Scaled dot-product attention with causal mask
        # scores: (B, n_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask (prevent attending to future tokens)
        # mask: (T, T) with -inf above diagonal
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device),
            diagonal=1
        )
        scores = scores + causal

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)                     # (B, n_heads, T, head_dim)

        # Back to (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        out = self.wo(out)
        out = self.resid_dropout(out)

        return out