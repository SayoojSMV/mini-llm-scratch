# finetune.py
import os
import torch
from torch.utils.data import DataLoader
from src.config import LLMConfig
from src.dataset import TextDataset
from src.tokenizer import Tokenizer
from src.model.transformer import MiniLLM

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Config & Dataset
    config = LLMConfig()
    tokenizer = Tokenizer()

    with open("data/instruct_data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    dataset = TextDataset(text=text, seq_len=config.max_seq_len, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)

    # 2. Load Base Model Weights
    model = MiniLLM(config).to(device)
    ckpt_path = "checkpoints/model.pt"

    if os.path.exists(ckpt_path):
        print(f"Loading pre-trained base model from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Unpack state_dict from dictionary wrapper
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

    # Lower learning rate (1e-4) for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    print("\n--- Starting Instruction Fine-Tuning ---")
    model.train()

    total_steps = 1500
    step = 0
    data_iter = iter(dataloader)

    while step < total_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits, loss = model(x, targets=y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % 100 == 0 or step == 1:
            print(f"Step {step:4d}/{total_steps} | Loss: {loss.item():.4f}")

    # Save fine-tuned model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/instruct_model.pt")
    print("\nFine-tuning complete! Saved to checkpoints/instruct_model.pt")

if __name__ == "__main__":
    main()