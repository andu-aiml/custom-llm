from dataclasses import dataclass

@dataclass
class LLMConfig:
    # tokenizer / data
    vocab_size: int
    context_len: int = 256

    # model
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2

    # ffn (SwiGLU)
    ffn_mult: float = 2.666

    # regularization
    dropout: float = 0.1

    # misc
    bias: bool = False
    rope_base: float = 10000.0  # (not used in your RoPE yet)