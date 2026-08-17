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
        # input_ids shape: [Batch_Size, Seq_Len]
        B, T = input_ids.shape
        
        # Positions: [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        
        # Combine token embeddings and positional embeddings
        tok_out = self.tok_embed(input_ids) # Shape: [B, T, d_model]
        pos_out = self.pos_embed(pos)       # Shape: [T, d_model]
        
        return tok_out + pos_out            # Shape: [B, T, d_model]