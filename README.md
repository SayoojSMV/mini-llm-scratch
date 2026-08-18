# mini-llm-scratch

A minimal, clean, and educational implementation of a GPT-style autoregressive Language Model built step-by-step from scratch using PyTorch.

## Architecture & Features
- **Tokenizer**: Byte-Pair Encoding wrapper around `tiktoken` (GPT-2 vocabulary).
- **Embeddings**: Token embeddings combined with learned positional embeddings.
- **Attention Mechanism**: Multi-Head Scaled Dot-Product Causal Self-Attention with efficient matrix operations and upper-triangular masking.
- **Normalization & Stability**: Pre-Layer Normalization (`Pre-LN`) topology with residual connections.
- **Non-Linearity**: Position-wise Feed-Forward Networks utilizing Gaussian Error Linear Units (`GELU`) with a $4\times d_{model}$ expansion ratio.

## Project Structure
```text
mini-llm-scratch/
├── src/
│   ├── config.py         # Model hyperparameter configuration
│   ├── tokenizer.py      # Tokenization pipeline wrapper
│   └── model/
│       ├── attention.py   # Multi-Head Causal Self-Attention
│       └── transformer.py # Embeddings and Feed-Forward Network
├── train.py              # Verification and training pipeline
├── README.md             # Project documentation
└── CHANGELOG.md          # Version history and module updates