"""
Full Transformer Architecture
==============================
Day 10 of Building Transformers from Scratch

Combines encoder and decoder into the complete Transformer
from "Attention is All You Need" (2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from encoder_stack import TransformerEncoder
from decoder_stack import TransformerDecoder
from embeddings import TransformerEmbedding


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.

    Architecture from the paper:
    - 6 encoder layers
    - 6 decoder layers
    - Weight sharing between embeddings and output projection
    - d_model = 512, n_heads = 8, d_ff = 2048
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000
    ):
        super().__init__()

        # Source side (encoder)
        self.src_embedding = TransformerEmbedding(
            src_vocab_size, d_model, max_seq_len, dropout
        )
        self.encoder = TransformerEncoder(
            n_layers, d_model, n_heads, d_ff, dropout, max_seq_len
        )

        # Target side (decoder)
        self.tgt_embedding = TransformerEmbedding(
            tgt_vocab_size, d_model, max_seq_len, dropout
        )
        self.decoder = TransformerDecoder(
            n_layers, d_model, n_heads, d_ff, dropout, max_seq_len
        )

        # Output projection (projects decoder output to vocabulary)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Weight sharing: Use embedding weights for output projection
        # This is a key detail from the paper!
        self.output_projection.weight = self.tgt_embedding.token_embedding.weight

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through full transformer.

        Args:
            src: Source token IDs [batch, src_len]
            tgt: Target token IDs [batch, tgt_len]
            src_mask: Source padding mask
            tgt_mask: Target padding mask

        Returns:
            Logits over target vocabulary [batch, tgt_len, tgt_vocab_size]
        """
        # Encode source sequence
        src_embedded = self.src_embedding(src)
        encoder_output = self.encoder(src_embedded, src_mask)

        # Decode target sequence (with encoder output for cross-attention)
        tgt_embedded = self.tgt_embedding(tgt)
        decoder_output = self.decoder(
            tgt_embedded, encoder_output, src_mask, tgt_mask
        )

        # Project to vocabulary size
        logits = self.output_projection(decoder_output)

        return logits

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """Encode source sequence (useful for inference)."""
        src_embedded = self.src_embedding(src)
        return self.encoder(src_embedded, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ):
        """Decode with given encoder output (useful for inference)."""
        tgt_embedded = self.tgt_embedding(tgt)
        decoder_output = self.decoder(
            tgt_embedded, encoder_output, src_mask, tgt_mask
        )
        return self.output_projection(decoder_output)


if __name__ == "__main__":
    print("="*60)
    print("FULL TRANSFORMER TEST")
    print("="*60)

    # Create model with paper's dimensions
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048
    )

    # Test input
    batch_size = 2
    src_len = 10
    tgt_len = 8

    src_tokens = torch.randint(0, 10000, (batch_size, src_len))
    tgt_tokens = torch.randint(0, 10000, (batch_size, tgt_len))

    # Forward pass
    logits = model(src_tokens, tgt_tokens)

    print(f"\nInput shapes:")
    print(f"  Source: {list(src_tokens.shape)}")
    print(f"  Target: {list(tgt_tokens.shape)}")
    print(f"\nOutput shape: {list(logits.shape)}")
    print(f"  Expected: [batch_size, tgt_len, vocab_size]")

    # Verify output
    assert logits.shape == (batch_size, tgt_len, 10000)
    print("\n✓ Shape check passed")

    # Test probability distribution
    probs = F.softmax(logits, dim=-1)
    print(f"\nProbability sum (should be 1.0): {probs[0, 0].sum().item():.4f}")

    print("\n" + "="*60)
    print("ARCHITECTURE SUMMARY")
    print("="*60)
    print("Source → Encoder (6 layers) → Encoder Output")
    print("Target → Decoder (6 layers + cross-attention) → Logits")
    print("Weight sharing: Embeddings ←→ Output Projection")