from dataclasses import dataclass

@dataclass
class LLMConfig:
    vocab_size: int = 50257   # GPT-2 tokenizer vocabulary size
    max_seq_len: int = 256    # Maximum sequence length (context window)
    d_model: int = 256        # Embedding vector dimension
    n_heads: int = 8
    n_layers: int = 4