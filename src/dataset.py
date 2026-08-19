# src/dataset.py
import urllib.request
import torch
from torch.utils.data import Dataset
from src.tokenizer import Tokenizer

class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, seq_len: int):
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        # Ensure we have enough remaining tokens for a full target shift
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        # Extract fixed-length input chunk and right-shifted target chunk
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_tinyshakespeare_data(filepath: str = "data/input.txt") -> str:
    """Downloads TinyShakespeare if not already downloaded and returns text content."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if not os.path.exists(filepath):
        print("Downloading TinyShakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, filepath)
        
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()