# train.py
import torch
from src.config import LLMConfig
from src.tokenizer import Tokenizer
from src.model.transformer import InputEmbeddings
from src.model.attention import MultiHeadAttention

# 1. Setup & Embeddings
config = LLMConfig()
tokenizer = Tokenizer()
text = "Building a Mini-LLM from scratch!"
token_ids = tokenizer.encode(text)
batch_ids = torch.stack([token_ids, token_ids], dim=0)

embedding_layer = InputEmbeddings(config)
x = embedding_layer(batch_ids)

# 2. Module 3 Multi-Head Attention Test
mha_layer = MultiHeadAttention(config)
mha_output = mha_layer(x)

print(f"Input Shape: {x.shape}")
print(f"MHA Output Shape: {mha_output.shape}")

assert mha_output.shape == x.shape
print("Module 3 Passed Successfully!")