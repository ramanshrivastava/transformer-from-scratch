"""
Decoder Stack for Transformer
==============================
Day 9 of Building Transformers from Scratch
Stacks 6 decoder blocks to create the full decoder from
"Attention is All You Need" (2017).
"""

import torch
import torch.nn as nn
from typing import Optional
from decoder_block import DecoderBlock
from attention_mechanism import PositionalEncoding


class TransformerDecoder(nn.Module):
    """
    Stack of N decoder layers, mirroring the encoder stack structure.

    From the paper: "The decoder is also composed of a stack of N = 6
    identical layers."
    """

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Positional encoding (same as encoder)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)

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
            encoder_output: From encoder [batch, src_len, d_model]
            src_mask: Source padding mask
            tgt_mask: Target padding mask

        Returns:
            [batch, tgt_len, d_model]
        """
        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # Final layer norm
        x = self.layer_norm(x)

        return x


if __name__ == "__main__":
    print("="*60)
    print("DECODER STACK TEST")
    print("="*60)

    # Create decoder stack
    decoder = TransformerDecoder(n_layers=6)

    # Test inputs
    batch_size = 2
    src_len = 10  # Source sequence
    tgt_len = 8   # Target sequence
    d_model = 512

    decoder_input = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)

    # Forward pass
    output = decoder(decoder_input, encoder_output)

    print(f"\nInput shapes:")
    print(f"  Decoder: {list(decoder_input.shape)}")
    print(f"  Encoder: {list(encoder_output.shape)}")
    print(f"Output: {list(output.shape)}")

    print(f"\nLayers: {decoder.n_layers}")
    print("✓ Decoder stack complete")