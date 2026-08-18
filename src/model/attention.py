# src/model/attention.py
import math
import torch
import torch.nn as nn
from src.config import LLMConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        # Combined QKV linear projection for efficiency
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Causal mask
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # C = d_model

        # 1. Compute Q, K, V in a single forward projection: [B, T, 3 * d_model]
        qkv = self.c_attn(x)
        
        # 2. Split Q, K, V: each becomes [B, T, d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. Reshape for Multi-Head: [B, T, d_model] -> [B, T, n_heads, d_k] -> [B, n_heads, T, d_k]
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 4. Scaled Dot-Product Attention: [B, n_heads, T, d_k] @ [B, n_heads, d_k, T] -> [B, n_heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 5. Apply causal mask
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # 6. Softmax & weighted aggregation
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # [B, n_heads, T, d_k]

        # 7. Concatenate heads: [B, n_heads, T, d_k] -> [B, T, n_heads, d_k] -> [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Output projection: [B, T, d_model]
        return self.c_proj(out)