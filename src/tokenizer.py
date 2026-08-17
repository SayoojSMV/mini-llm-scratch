import tiktoken
import torch

class Tokenizer:
    def __init__(self, model_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(model_name)

    def encode(self, text: str) -> torch.Tensor:
        tokens = self.enc.encode(text)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return self.enc.decode(tokens.tolist())