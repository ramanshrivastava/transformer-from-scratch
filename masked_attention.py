"""
Masked Multi-Head Attention for Transformer Decoders
=====================================================
Extends our Day 1 MultiHeadAttention with causal masking
to enable autoregressive text generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from attention_mechanism import MultiHeadAttention


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create a causal mask to prevent attending to future positions.

    Args:
        seq_len: Length of the sequence

    Returns:
        Lower triangular matrix (1 = can attend, 0 = cannot attend)
    """
    return torch.tril(torch.ones(seq_len, seq_len))


class MaskedMultiHeadAttention(MultiHeadAttention):
    """
    Multi-Head Attention with causal masking for decoders.
    Inherits everything from MultiHeadAttention, just adds masking.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal masking.

        Args:
            query, key, value: [batch_size, seq_len, d_model]
            mask: Optional padding mask

        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len = query.shape[:2]

        # Project Q, K, V (using parent's layers)
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply causal mask - THE ONLY DIFFERENCE FROM ENCODER
        causal_mask = create_causal_mask(seq_len).to(scores.device)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Apply padding mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply to values and reshape
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output, attention_weights


if __name__ == "__main__":
    # Simple demonstration
    print("="*60)
    print("CAUSAL MASK PATTERN")
    print("="*60)

    # Show what the mask looks like
    mask = create_causal_mask(4)
    print("\nFor sequence length 4:")
    print(mask.int())
    print("\n✓ = can attend (1), ✗ = blocked (0)")
    print("Each position can only attend to itself and earlier positions")

    # Test it works
    print("\n" + "="*60)
    print("TESTING MASKED ATTENTION")
    print("="*60)

    masked_attn = MaskedMultiHeadAttention(d_model=512, n_heads=8)
    x = torch.randn(2, 10, 512)  # [batch=2, seq=10, d_model=512]

    output, weights = masked_attn(x, x, x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Verify masking worked
    print("\nChecking first attention head (position 2 should not see position 3+):")
    attn_sample = weights[0, 0, 2, :5]  # Batch 0, Head 0, Position 2
    print(f"Position 2 attention to positions 0-4: {attn_sample.detach().numpy().round(3)}")
    print("(Should be ~0 for positions 3-4)")