"""
Decoder Block for Transformer
==============================
Implements the decoder block from "Attention is All You Need" (2017).
Three sublayers: masked self-attention, cross-attention, feed-forward.
"""

import torch
import torch.nn as nn
from typing import Optional

from masked_attention import MaskedMultiHeadAttention
from attention_mechanism import MultiHeadAttention


class DecoderBlock(nn.Module):
    """
    Decoder block with 3 sublayers:
    1. Masked self-attention (causal)
    2. Cross-attention (decoder → encoder)
    3. Feed-forward network
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        # Sublayer 1: Masked self-attention
        self.masked_attention = MaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Sublayer 2: Cross-attention (same as MultiHeadAttention)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Sublayer 3: Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input [batch, tgt_len, d_model]
            encoder_output: Encoder output [batch, src_len, d_model]
            src_mask: Source padding mask
            tgt_mask: Target padding mask

        Returns:
            [batch, tgt_len, d_model]
        """
        # 1. Masked self-attention
        attn_out, _ = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Cross-attention: Q from decoder, K,V from encoder
        cross_out, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # 3. Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x


if __name__ == "__main__":
    print("="*60)
    print("DECODER BLOCK TEST")
    print("="*60)

    # Parameters from paper
    decoder = DecoderBlock(d_model=512, n_heads=8, d_ff=2048)

    # Test shapes
    batch, src_len, tgt_len, d_model = 2, 10, 8, 512
    decoder_input = torch.randn(batch, tgt_len, d_model)
    encoder_output = torch.randn(batch, src_len, d_model)

    output = decoder(decoder_input, encoder_output)

    print(f"\nDecoder input:  {list(decoder_input.shape)}")
    print(f"Encoder output: {list(encoder_output.shape)}")
    print(f"Output:         {list(output.shape)}")

    assert output.shape == decoder_input.shape
    print("\n✓ Shapes correct")

    # Parameter count
    params = sum(p.numel() for p in decoder.parameters())
    print(f"\nParameters: {params:,}")