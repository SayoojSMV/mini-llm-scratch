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
- **Dataset Pipeline**: TextDataset chunking for autoregressive target-shifting with dataset preparation scripts.
- **Instruction Tuning & Hardware Acceleration**: Supervised fine-tuning pipeline utilizing PyTorch AMP (`torch.amp`) for FP16 accelerated execution.
- **Persistence & CLI**: State dictionary checkpoint saving/loading and an interactive conversational CLI interface (`generate.py`).

## Project Structure
mini-llm-scratch/
├── checkpoints/              # Saved base and instruct model checkpoints
├── data/                     # Raw and processed dataset files
├── src/
│   ├── config.py             # Model hyperparameter configuration
│   ├── dataset.py            # PyTorch Dataset and auto-downloader
│   ├── prepare_instruct_data.py # Alpaca dataset formatting utility
│   ├── tokenizer.py          # Tokenization pipeline wrapper
│   └── model/
│       ├── attention.py      # Multi-Head Causal Self-Attention
│       └── transformer.py    # Full MiniLLM architecture & generation
├── finetune.py               # Instruction fine-tuning pipeline (AMP optimized)
├── generate.py               # Interactive CLI generation interface
├── train.py                  # Base pre-training pipeline
├── README.md                 # Project documentation
└── CHANGELOG.md              # Version history and module updates

## Quick Start

### 1. Base Pre-Training
Train MiniLLM on raw text to build base language modeling capabilities:

python train.py

### 2. Instruction Fine-Tuning
Format the instruction dataset and fine-tune your base checkpoint into a chat assistant:

python src/prepare_instruct_data.py
python finetune.py

### 3. Interactive Generation
Launch the interactive CLI to prompt your instruction-tuned model:

python generate.py