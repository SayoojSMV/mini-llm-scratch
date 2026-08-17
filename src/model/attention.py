# src/model/attention.py
import math
import torch
import torch.nn as nn
from src.config import LLMConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.d_k = config.d_model
        
        # Linear projections for Query, Key, Value
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Causal mask: Lower triangular matrix of ones
        # register_buffer ensures mask is saved with module state without being a parameter
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d_model = x.shape
        
        # Compute Q, K, V projections
        Q = self.w_q(x) # [B, T, d_model]
        K = self.w_k(x) # [B, T, d_model]
        V = self.w_v(x) # [B, T, d_model]
        
        # Compute scaled attention scores: [B, T, d_model] @ [B, d_model, T] -> [B, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask: replace upper-triangular zeros with -infinity
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        
        # Softmax along sequence dimension to get attention weights
        attn_weights = torch.softmax(scores, dim=-1) # [B, T, T]
        
        # Weighted sum over values: [B, T, T] @ [B, T, d_model] -> [B, T, d_model]
        output = torch.matmul(attn_weights, V)
        return output