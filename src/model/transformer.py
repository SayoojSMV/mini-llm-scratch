# src/model/transformer.py
import torch
import torch.nn as nn
from src.config import LLMConfig

class InputEmbeddings(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_out = self.tok_embed(input_ids)
        pos_out = self.pos_embed(pos)
        return tok_out + pos_out


class FeedForward(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        # Expand dimension by 4x inside the hidden layer
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, d_model] -> [B, T, 4 * d_model] -> [B, T, d_model]
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x