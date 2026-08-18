# train.py
import torch
import torch.nn as nn
from src.config import LLMConfig
from src.tokenizer import Tokenizer
from src.model.transformer import InputEmbeddings, FeedForward
from src.model.attention import MultiHeadAttention

# 1. Setup & Embeddings
config = LLMConfig()
tokenizer = Tokenizer()
text = "Building a Mini-LLM from scratch!"
token_ids = tokenizer.encode(text)
batch_ids = torch.stack([token_ids, token_ids], dim=0)

embedding_layer = InputEmbeddings(config)
x = embedding_layer(batch_ids)

# 2. Sub-layer 1: Pre-LN Multi-Head Attention + Residual
ln_1 = nn.LayerNorm(config.d_model)
mha = MultiHeadAttention(config)
x_attn = x + mha(ln_1(x))

# 3. Sub-layer 2: Pre-LN FeedForward + Residual
ln_2 = nn.LayerNorm(config.d_model)
ffn = FeedForward(config)
x_out = x_attn + ffn(ln_2(x_attn))

print(f"Input Shape: {x.shape}")
print(f"Sub-layer Output Shape: {x_out.shape}")

assert x_out.shape == x.shape
print("Module 4 Passed Successfully!")