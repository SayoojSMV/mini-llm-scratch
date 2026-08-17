import torch
from src.config import LLMConfig
from src.tokenizer import Tokenizer
from src.model.transformer import InputEmbeddings

# Initialize setup
config = LLMConfig()
tokenizer = Tokenizer()

# Tokenize test sequence
text = "Building a Mini-LLM from scratch!"
token_ids = tokenizer.encode(text)

# Create a batch of 2 identical sequences
batch_ids = torch.stack([token_ids, token_ids], dim=0) # [B, T] = [2, T]

# Run through embedding layer
embedding_layer = InputEmbeddings(config)
embedded_output = embedding_layer(batch_ids)

print(f"Token IDs shape: {token_ids.shape}")
print(f"Batch Input shape: {batch_ids.shape}")
print(f"Embedded Output shape: {embedded_output.shape}")

assert embedded_output.shape == (2, token_ids.shape[0], config.d_model)
print("Module 1 Passed Successfully!")