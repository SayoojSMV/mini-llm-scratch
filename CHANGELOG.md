# Changelog

All notable changes to the mini-llm-scratch project will be documented in this file.

## [0.10.0] - Module 10 Complete (Retrieval-Augmented Generation)
### Added
- Dense vector RAG retriever (`src/rag/retriever.py`) powered by `sentence-transformers` (`all-MiniLM-L6-v2`) and cosine similarity.
- Local document indexing system targeting `data/knowledge_base/*.txt` chunks with relevance score thresholding (>0.25).

### Changed
- Updated `generate.py` to automatically retrieve top-k context snippets and inline them into `User:` prompts for zero-hallucination inference.
- Lowered sampling temperature to `0.3` in RAG generation mode for strict factual context adherence.

## [0.9.0] - Module 9 Complete (Instruction Fine-Tuning & Performance Optimization)
### Added
- Instruction fine-tuning pipeline (`finetune.py`) to transition the base model into an interactive instruction-following assistant.
- Dataset processing script (`src/prepare_instruct_data.py`) to convert raw Alpaca JSON instruction sets into formatted `User:` / `Assistant:` prompt pairs.
- Automatic Mixed Precision (`torch.amp`) integration across training and fine-tuning scripts to accelerate CUDA matrix calculations and lower VRAM consumption.
- Asynchronous DataLoader features (`pin_memory=True`, `non_blocking=True`) to eliminate host-to-device memory transfer bottlenecks.

## [0.8.0] - Module 8 Complete
### Added
- Checkpoint persistence utilities (save_checkpoint and load_checkpoint) in MiniLLM.
- Interactive command-line generation interface (generate.py).

## [0.7.0] - Module 7 Complete
### Added
- TextDataset class in src/dataset.py for input/target context window chunking.
- Auto-download utility for the TinyShakespeare dataset.
- Training loop in train.py with AdamW optimization.

## [0.6.0] - Module 6 Complete
### Added
- Autoregressive generate method in MiniLLM.
- Decoding sampling strategies: Greedy, Temperature scaling, and Top-k filtering.

## [0.5.0] - Module 5 Complete
### Added
- Complete MiniLLM transformer stack combining blocks and final LayerNorm.
- Linear LM Head with weight tying to token embeddings.
- Cross-entropy loss calculation interface for target tokens.

## [0.4.0] - Module 4 Complete
### Added
- Position-wise FeedForward network module with 4x d_model expansion and GELU activation.
- LayerNorm integration verifying Pre-LN residual pipeline in train.py.

## [0.3.0] - Module 3 Complete
### Added
- MultiHeadAttention class using unified Q, K, V linear projections (c_attn).
- Scaled dot-product attention parallelized across h=8 heads.
- Updated LLMConfig to support multi-head hyperparameters.

## [0.2.0] - Module 2 Complete
### Added
- Single-head CausalSelfAttention with lower-triangular causal masking (torch.tril).
- Scaled dot-product normalization factor.

## [0.1.0] - Module 1 Complete
### Added
- Project structure setup (src/, config.py, tokenizer.py).
- Token embedding and learned positional embedding layer (InputEmbeddings).
- BPE Tokenizer integration using tiktoken.