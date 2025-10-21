"""
Attention Is All You Need - Simple & Elegant Implementation
=============================================================
A clean implementation of the Transformer attention mechanism from the paper:
"Attention Is All You Need" (Vaswani et al., 2017)

Author: Implementation based on the original paper https://arxiv.org/abs/1706.03762
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(QK^T / √d_k)V

    Where:
        - Q: Query matrix (what information am I looking for?)
        - K: Key matrix (what information do I have?)
        - V: Value matrix (what is the actual information?)
        - d_k: Dimension of the key vectors (for scaling)
    """

    def __init__(self, temperature=1.0, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, n_heads, seq_len, d_k]
            key:   [batch_size, n_heads, seq_len, d_k]
            value: [batch_size, n_heads, seq_len, d_v]
            mask:  [batch_size, 1, seq_len, seq_len] (optional)

        Returns:
            output: [batch_size, n_heads, seq_len, d_v]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, n_heads, seq_len, d_k = query.shape

        # Step 1: Compute attention scores
        # QK^T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Step 2: Apply mask (if provided) - for padding or causal attention
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Step 3: Apply softmax to get attention weights
        attention_weights = F.softmax(scores / self.temperature, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Step 4: Apply attention weights to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    Instead of performing a single attention function, we found it beneficial
    to linearly project the queries, keys and values h times with different,
    learned linear projections to d_k, d_k and d_v dimensions, respectively.
    """

    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        
        """
        Args:
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.d_v = d_model // n_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key:   [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask:  [batch_size, seq_len, seq_len] (optional)

        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape

        # Step 1: Linear projections in batch from d_model => n_heads x d_k
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)

        # Step 2: Apply scaled dot-product attention
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension

        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Step 3: Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.W_o(attn_output)

        return output, attn_weights


class PositionalEncoding(nn.Module):
    """
    Positional Encoding

    Since our model contains no recurrence and no convolution, we must inject
    some information about the relative or absolute position of the tokens.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model=512, max_seq_len=5000):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerBlock(nn.Module):
    """
    A simple Transformer Block combining:
    1. Multi-Head Attention
    2. Layer Normalization
    3. Feed Forward Network
    4. Residual Connections
    """

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] (optional)

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-Attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed Forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output


def create_padding_mask(seq, pad_idx=0):
    """Create a mask to hide padding tokens"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len):
    """Create a mask for causal (autoregressive) attention"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0


# ============================================================================
# Example Usage & Demonstration
# ============================================================================

def demonstrate_attention():
    """
    Demonstrate the attention mechanism with a simple example
    """
    print("=" * 70)
    print("TRANSFORMER ATTENTION MECHANISM DEMONSTRATION")
    print("=" * 70)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Example parameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8

    print(f"\nConfiguration:")
    print(f"  • Batch Size: {batch_size}")
    print(f"  • Sequence Length: {seq_len}")
    print(f"  • Model Dimension: {d_model}")
    print(f"  • Number of Heads: {n_heads}")

    # Create sample input (e.g., word embeddings)
    input_embeddings = torch.randn(batch_size, seq_len, d_model)

    # Add positional encoding
    pos_encoder = PositionalEncoding(d_model)
    input_with_pos = pos_encoder(input_embeddings)

    # Create transformer block
    transformer_block = TransformerBlock(d_model, n_heads)

    # Forward pass
    output = transformer_block(input_with_pos)

    print(f"\nInput Shape:  {input_embeddings.shape}")
    print(f"Output Shape: {output.shape}")

    # Demonstrate attention weights
    attention = MultiHeadAttention(d_model, n_heads)
    _, attention_weights = attention(input_with_pos, input_with_pos, input_with_pos)

    print(f"\nAttention Weights Shape: {attention_weights.shape}")
    print("  (batch_size, n_heads, seq_len, seq_len)")

    # Show sample attention pattern for first head of first batch
    sample_attention = attention_weights[0, 0].detach().numpy()

    print("\nSample Attention Pattern (First 5x5):")
    print("  " + "  ".join([f"T{i:2d}" for i in range(5)]))
    for i in range(5):
        print(f"T{i:2d}", end=" ")
        for j in range(5):
            print(f"{sample_attention[i, j]:.2f}", end=" ")
        print()

    print("\n" + "=" * 70)
    print("Key Insights from 'Attention Is All You Need':")
    print("=" * 70)
    print("""
1. Self-Attention allows the model to look at all positions in the input
   sequence simultaneously, capturing long-range dependencies.

2. Multi-Head Attention allows the model to jointly attend to information
   from different representation subspaces at different positions.

3. Positional Encoding adds position information since attention itself
   is permutation-invariant.

4. The architecture is highly parallelizable, making it much faster to
   train than RNNs or LSTMs.

5. This simple mechanism forms the foundation of modern LLMs like
   GPT, BERT, and many others.
    """)


if __name__ == "__main__":
    demonstrate_attention()

    print("\nTo use in your own code:")
    print("-" * 40)
    print("""
# Initialize components
attention = MultiHeadAttention(d_model=512, n_heads=8)
transformer = TransformerBlock(d_model=512, n_heads=8)

# Process sequences
output = transformer(your_embeddings)
    """)