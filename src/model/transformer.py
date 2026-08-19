import torch
import torch.nn as nn
from src.config import LLMConfig
from src.model.attention import MultiHeadAttention

class InputEmbeddings(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        return self.tok_embed(input_ids) + self.pos_embed(pos)


class FeedForward(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.embeddings = InputEmbeddings(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between embedding and final linear projection
        self.embeddings.tok_embed.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        x = self.embeddings(input_ids)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Flatten predictions and targets to compute cross-entropy loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 20, 
        temperature: float = 1.0, 
        top_k: int = None
    ) -> torch.Tensor:
        """
        Autoregressively generates new tokens given a context sequence.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context sequence if it exceeds the model's maximum context length
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            
            # 1. Forward model to get logits
            logits, _ = self(idx_cond)
            
            # 2. Focus only on the last position: [B, vocab_size]
            logits = logits[:, -1, :]
            
            # 3. Apply Temperature scaling
            if temperature > 0:
                logits = logits / temperature
            
            # 4. Apply Top-k filtering (optional)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Mask out any logit smaller than the top-k threshold
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # 5. Convert logits to probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            
            # 6. Sample next token index
            if temperature == 0:
                # Greedy decoding
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                # Multinomial sampling
                idx_next = torch.multinomial(probs, num_samples=1)
                
            # 7. Append sampled index to sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids