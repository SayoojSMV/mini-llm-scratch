# train.py
import torch
from src.config import LLMConfig
from src.tokenizer import Tokenizer
from src.model.transformer import MiniLLM

# 1. Setup Model & Config
config = LLMConfig()
model = MiniLLM(config)
tokenizer = Tokenizer()

# 2. Prepare Data
text = "Building a Mini-LLM from scratch!"
token_ids = tokenizer.encode(text)
input_ids = torch.stack([token_ids, token_ids], dim=0)

# Shift targets right for next-token prediction test
targets = input_ids.clone()

# 3. Forward Pass
logits, loss = model(input_ids, targets=targets)

# Total parameters calculation
num_params = sum(p.numel() for p in model.parameters())

print(f"Input Shape: {input_ids.shape}")
print(f"Logits Shape: {logits.shape}")
print(f"Loss Value: {loss.item():.4f}")
print(f"Total Parameters: {num_params:,}")

assert logits.shape == (2, input_ids.shape[1], config.vocab_size)
print("Module 5 Passed Successfully!")