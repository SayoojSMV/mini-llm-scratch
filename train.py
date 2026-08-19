# train.py
import torch
from torch.utils.data import DataLoader
from src.config import LLMConfig
from src.tokenizer import Tokenizer
from src.dataset import TextDataset, get_tinyshakespeare_data
from src.model.transformer import MiniLLM

# 1. Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Config & Tokenizer
config = LLMConfig()
tokenizer = Tokenizer()

# 3. Data Loading
raw_text = get_tinyshakespeare_data()
dataset = TextDataset(raw_text, tokenizer, seq_len=config.max_seq_len)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 4. Model & Optimizer Setup
model = MiniLLM(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 5. Pre-Training Prompt Verification
prompt = "First Citizen:"
prompt_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)

print("\n--- Model Output Before Training ---")
before_ids = model.generate(prompt_ids.clone(), max_new_tokens=20, temperature=0.7)
print(f"Generated: {tokenizer.decode(before_ids[0])}\n")

# 6. Training Loop (100 Iterations)
print("--- Starting Training Loop ---")
model.train()
data_iter = iter(dataloader)

for step in range(1, 101):
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        x, y = next(data_iter)

    x, y = x.to(device), y.to(device)

    # Forward pass
    logits, loss = model(x, targets=y)

    # Backward pass & Optimizer Step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0 or step == 1:
        print(f"Step {step:3d}/100 | Loss: {loss.item():.4f}")

# 7. Post-Training Prompt Verification
print("\n--- Model Output After Training (100 Steps) ---")
model.eval()
after_ids = model.generate(prompt_ids.clone(), max_new_tokens=20, temperature=0.7)
print(f"Generated: {tokenizer.decode(after_ids[0])}")

print("\nModule 7 Passed Successfully!")