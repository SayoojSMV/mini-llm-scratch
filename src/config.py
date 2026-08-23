# src/config.py
from dataclasses import dataclass

@dataclass
class LLMConfig:
    vocab_size: int = 50257
    max_seq_len: int = 256     # Context window (was 128)
    d_model: int = 384         # Model embedding dimension (was 128)
    n_heads: int = 6           # Attention heads (384 / 6 = 64 head dim)
    n_layers: int = 6          # Number of Transformer blocks (was 2)
    dropout: float = 0.1