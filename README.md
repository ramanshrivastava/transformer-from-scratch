# Transformer Architecture: Attention Is All You Need

A clean, educational implementation of the Transformer architecture from the groundbreaking paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

## 🎯 Core Equation

The heart of the transformer architecture - Scaled Dot-Product Attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- **Q** (Query): What information am I looking for?
- **K** (Key): What information do I have?
- **V** (Value): What is the actual information?
- **d_k**: Dimension of the key vectors (for scaling)

## 📊 Why This Matters

This single equation:
- Powers ChatGPT, Claude, BERT, and virtually all modern LLMs
- Is the most GPU-intensive operation in transformers (O(n²) complexity)
- Explains why we have context length limits
- Revolutionized NLP and AI

## 🚀 Features

- ✅ **Scaled Dot-Product Attention**: The fundamental attention mechanism
- ✅ **Multi-Head Attention**: Parallel attention mechanisms for richer representations
- ✅ **Positional Encoding**: Sinusoidal position embeddings
- ✅ **Transformer Block**: Complete encoder block with residual connections
- ✅ **Visualizations**: Beautiful attention pattern visualizations

## 📁 Files

- `attention_mechanism.py` - Core implementation of attention and transformer components
- `visualize_attention.py` - Visualization tools for understanding attention patterns
- `attention_is_all_you_need.pdf` - The original paper

## 🔥 Quick Start

```python
from attention_mechanism import ScaledDotProductAttention, MultiHeadAttention

# Initialize attention
attention = MultiHeadAttention(d_model=512, n_heads=8)

# Process sequences
output, attention_weights = attention(query, key, value)
```

## 💡 Key Insights

1. **Self-Attention** allows the model to look at all positions in the input sequence simultaneously
2. **Multi-Head Attention** learns different types of relationships in parallel
3. **Positional Encoding** preserves sequence order information
4. **Quadratic Complexity** (O(n²)) is why context windows are limited

## 📈 Computational Complexity

For a sequence of length n and dimension d:
- Attention: **O(n²·d)** operations
- Memory: **O(n²)** to store attention matrix
- Example: For n=2048, d=512 → ~1 billion operations!

## 🔗 Connect

Created as an educational implementation to understand the architecture that revolutionized AI.

---

*"Attention is all you need" - and this equation proves it.*