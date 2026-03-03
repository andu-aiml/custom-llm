# 🧠 LLaMA-Style LLM From Scratch (PyTorch)

A modern decoder-only Large Language Model implemented fully from scratch using PyTorch.

This project implements a LLaMA-style architecture with:

- Byte Pair Encoding (BPE) tokenizer
- Rotary Positional Embedding (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU Feed-Forward Network
- RMSNorm normalization
- Pre-Norm Transformer blocks
- Causal Language Modeling objective

Trained on WikiText-2 using CPU.

---

## 🚀 Architecture Overview

- Decoder-only Transformer
- Pre-Norm residual connections
- Weight tying between embedding and LM head
- Causal masking
- AdamW optimizer
- Gradient clipping

---

## 📦 Model Configuration (Default)

| Parameter | Value |
|-----------|--------|
| d_model | 384 |
| n_layers | 8 |
| n_heads | 8 |
| n_kv_heads | 2 |
| context_length | 256 |
| ffn_multiplier | 2.666 |
| dropout | 0.1 |

---

## 📊 Dataset

- WikiText-2 (raw)
- Custom BPE tokenizer trained from scratch
- Entire corpus tokenized into contiguous tensor

---

## 🏗 Modules Implemented

### ✔ RMSNorm
Root Mean Square Layer Normalization (LLaMA-style)

### ✔ RoPE
Rotary Positional Embedding applied to Q and K

### ✔ GQA
Grouped Query Attention for memory efficiency

### ✔ SwiGLU
Gated feed-forward network using SiLU activation

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/llm-from-scratch.git
cd llm-from-scratch
pip install -r requirements.txt
