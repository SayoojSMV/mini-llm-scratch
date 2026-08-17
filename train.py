# train.py
import torch
from src.config import LLMConfig
from src.tokenizer import Tokenizer
from src.model.transformer import InputEmbeddings
from src.model.attention import CausalSelfAttention

# 1. Setup & Embeddings
config = LLMConfig()
tokenizer = Tokenizer()
text = "Building a Mini-LLM from scratch!"
token_ids = tokenizer.encode(text)
batch_ids = torch.stack([token_ids, token_ids], dim=0)

embedding_layer = InputEmbeddings(config)
x = embedding_layer(batch_ids)

# 2. Causal Self Attention Test
attention_layer = CausalSelfAttention(config)
attn_output = attention_layer(x)

print(f"Input Shape: {x.shape}")
print(f"Attention Output Shape: {attn_output.shape}")

assert attn_output.shape == x.shape
print("Module 2 Passed Successfully!")