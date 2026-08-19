# train.py
import torch
from src.config import LLMConfig
from src.tokenizer import Tokenizer
from src.model.transformer import MiniLLM

# 1. Setup Model & Config
config = LLMConfig()
model = MiniLLM(config)
tokenizer = Tokenizer()

# 2. Prepare Prompt
prompt_text = "Building a"
input_ids = tokenizer.encode(prompt_text).unsqueeze(0) # [1, T]

# 3. Test Generation with Sampling Strategies
print("\n--- Testing Text Generation ---")

# Strategy A: Greedy Generation
greedy_ids = model.generate(input_ids.clone(), max_new_tokens=15, temperature=0)
print(f"\n[Greedy (Temp=0)]: {tokenizer.decode(greedy_ids[0])}")

# Strategy B: Temperature Sampling (Temp=0.8, Top-k=10)
sampled_ids = model.generate(input_ids.clone(), max_new_tokens=15, temperature=0.8, top_k=10)
print(f"\n[Sampled (Temp=0.8, Top-k=10)]: {tokenizer.decode(sampled_ids[0])}")

print("\nModule 6 Passed Successfully!")