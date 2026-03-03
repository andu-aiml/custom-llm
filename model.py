import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LLMConfig
from rms_norm import RMSNorm
from gqa import GQAAttention
from swiglu import SwiGLUFeedForward
from rope import RoPE

# ---- your modules (use your existing ones) ----
# RMSNorm, RoPE, SwiGLUFeedForward, GQAAttention should be imported or pasted above.


class TransformerBlock(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = GQAAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            rope=rope,
            dropout=cfg.dropout,
            bias=cfg.bias
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFeedForward(
            d_model=cfg.d_model,
            hidden_mult=cfg.ffn_mult,
            dropout=cfg.dropout,
            bias=cfg.bias
        )

    def forward(self, x):
        # Pre-norm + residual
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderOnlyLM(nn.Module):
    def __init__(self, cfg, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_id)
        self.drop = nn.Dropout(cfg.dropout)

        head_dim = cfg.d_model // cfg.n_heads
        self.rope = RoPE(dim=head_dim, max_seq_len=cfg.context_len)

        self.blocks = nn.ModuleList([TransformerBlock(cfg, self.rope) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying (common trick)
        self.lm_head.weight = self.tok_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token ids
        targets: (B, T) token ids (next-token)
        """
        B, T = idx.shape
        assert T <= self.cfg.context_len, "Sequence length exceeds context_len"

        x = self.tok_embed(idx)          # (B,T,d_model)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)         # (B,T,vocab)

        loss = None
        if targets is not None:
            # standard next-token CE
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.pad_id
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        idx: (B, T)
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_len:]  # crop to context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx