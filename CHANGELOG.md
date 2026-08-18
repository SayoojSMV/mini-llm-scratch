# Changelog

All notable changes to the `mini-llm-scratch` project will be documented in this file.

## [0.4.0] - Module 4 Complete
### Added
- Position-wise `FeedForward` network module with $4\times d_{model}$ expansion and `GELU` activation.
- `LayerNorm` integration verifying Pre-LN residual pipeline in `train.py`.

## [0.3.0] - Module 3 Complete
### Added
- `MultiHeadAttention` class using unified $Q, K, V$ linear projections (`c_attn`).
- Scaled dot-product attention parallelized across $h=8$ heads.
- Updated `LLMConfig` to support multi-head hyperparameters.

## [0.2.0] - Module 2 Complete
### Added
- Single-head `CausalSelfAttention` with lower-triangular causal masking (`torch.tril`).
- Scaled dot-product normalization factor ($\frac{1}{\sqrt{d_k}}$).

## [0.1.0] - Module 1 Complete
### Added
- Project structure setup (`src/`, `config.py`, `tokenizer.py`).
- Token embedding and learned positional embedding layer (`InputEmbeddings`).
- BPE Tokenizer integration using `tiktoken`.