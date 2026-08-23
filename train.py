# train.py
import os
import urllib.request
import torch
from torch.utils.data import DataLoader
from src.config import LLMConfig
from src.dataset import TextDataset
from src.tokenizer import Tokenizer
from src.model.transformer import MiniLLM

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Setup Config & Tokenizer
    config = LLMConfig()
    tokenizer = Tokenizer()
    
    # 2. Ensure Dataset Exists
    os.makedirs("data", exist_ok=True)
    data_path = os.path.join("data", "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 3. High-Speed DataLoader
    dataset = TextDataset(text=text, seq_len=config.max_seq_len, tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=16,          # Optimal batch size for 4GB VRAM
        shuffle=True, 
        pin_memory=True,        # Enables fast pinned memory transfer
        num_workers=2           # Multi-threaded data loading
    )

    # 4. Model, Optimizer, and FP16 Scaler
    model = MiniLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() # FP16 Mixed Precision

    print("\n--- Starting High-Speed Training Loop ---")
    model.train()
    
    total_steps = 3000
    step = 0
    data_iter = iter(dataloader)

    while step < total_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        # Asynchronous VRAM transfer
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # FP16 Forward Pass
        with torch.cuda.amp.autocast():
            logits, loss = model(x, targets=y)
        
        # Backward Pass with Scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % 200 == 0 or step == 1:
            print(f"Step {step:4d}/{total_steps} | Loss: {loss.item():.4f}")

    # 5. Save Model
    os.makedirs("checkpoints", exist_ok=True)
    model.save_checkpoint("checkpoints/model.pt")
    print("\nTraining Complete! Checkpoint saved.")

if __name__ == "__main__":
    main()