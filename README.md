# mini-llm-scratch

A minimal, clean, and educational implementation of a GPT-style autoregressive Language Model built step-by-step from scratch using PyTorch.

## Architecture & Features
- **Tokenizer**: Byte-Pair Encoding wrapper around tiktoken (GPT-2 vocabulary).
- **Embeddings**: Token embeddings combined with learned positional embeddings.
- **Attention Mechanism**: Multi-Head Scaled Dot-Product Causal Self-Attention with efficient matrix operations and upper-triangular masking.
- **Normalization & Stability**: Pre-Layer Normalization (Pre-LN) topology with residual connections.
- **Non-Linearity**: Position-wise Feed-Forward Networks utilizing GELU with a 4x d_model expansion ratio.
- **Weight Tying**: Shared parameter weights between token embeddings and the final LM output head.
- **Inference & Sampling**: Autoregressive decoding loop supporting Greedy search, Temperature scaling, and Top-k filtering.
- **Dataset Pipeline**: TextDataset chunking for autoregressive target-shifting with auto-download for TinyShakespeare.
- **Persistence & CLI**: State dictionary checkpoint saving/loading and an interactive CLI interface (generate.py).

## Project Structure
mini-llm-scratch/
├── checkpoints/          # Model state dictionaries
├── data/                 # Training dataset storage
├── src/
│   ├── config.py         # Model hyperparameter configuration
│   ├── dataset.py        # PyTorch Dataset and auto-downloader
│   ├── tokenizer.py      # Tokenization pipeline wrapper
│   └── model/
│       ├── attention.py   # Multi-Head Causal Self-Attention
│       └── transformer.py # Full MiniLLM architecture & generation
├── generate.py           # Interactive CLI generation interface
├── train.py              # Verification and training pipeline
├── README.md             # Project documentation
└── CHANGELOG.md          # Version history and module updates

## Quick Start

### 1. Training
Run the training loop to train MiniLLM on TinyShakespeare and save a checkpoint:

python train.py

### 2. Interactive Generation
Launch the interactive CLI to prompt your trained model:

python generate.py