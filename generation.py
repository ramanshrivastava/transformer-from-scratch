"""
Text Generation for Transformer
================================
Day 11 of Building Transformers from Scratch

Implements autoregressive generation as described in
"Attention is All You Need" (2017), including beam search.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from transformer import Transformer
from masked_attention import create_causal_mask


class TransformerGenerator:
    """
    Generator for autoregressive text generation.
    Implements both greedy and beam search decoding.
    """

    def __init__(self, model: Transformer, max_length: int = 100):
        self.model = model
        self.max_length = max_length

    @torch.no_grad()
    def greedy_generate(
        self,
        src: torch.Tensor,
        start_token: int,
        end_token: int,
        temperature: float = 1.0
    ) -> List[int]:
        """
        Greedy generation: select highest probability token at each step.

        Args:
            src: Source tokens [1, src_len]
            start_token: Token to start generation
            end_token: Token to stop generation
            temperature: Sampling temperature (1.0 = no change)

        Returns:
            Generated token sequence
        """
        self.model.eval()

        # Encode source once (key insight from paper!)
        encoder_output = self.model.encode(src)

        # Start with [START] token
        tgt = torch.tensor([[start_token]])

        for _ in range(self.max_length):
            # Create causal mask for decoder
            tgt_mask = create_causal_mask(tgt.size(1))

            # Decode with current sequence
            logits = self.model.decode(
                tgt, encoder_output, tgt_mask=tgt_mask
            )

            # Get probabilities for last position
            logits_last = logits[:, -1, :] / temperature
            probs = F.softmax(logits_last, dim=-1)

            # Select highest probability token
            next_token = torch.argmax(probs, dim=-1).item()

            # Stop if we hit end token
            if next_token == end_token:
                break

            # Append and continue
            tgt = torch.cat([tgt, torch.tensor([[next_token]])], dim=1)

        return tgt[0].tolist()

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        start_token: int,
        end_token: int,
        beam_size: int = 4,
        length_penalty: float = 0.6
    ) -> List[int]:
        """
        Beam search as described in the paper (Section 5.3).
        Uses beam size of 4 and length penalty α = 0.6.

        Args:
            src: Source tokens [1, src_len]
            start_token: Token to start generation
            end_token: Token to stop generation
            beam_size: Number of beams (paper uses 4)
            length_penalty: Length normalization (paper uses 0.6)

        Returns:
            Best sequence from beam search
        """
        self.model.eval()

        # Encode source once
        encoder_output = self.model.encode(src)

        # Initialize beams: (score, sequence)
        beams = [(0.0, [start_token])]
        complete_sequences = []

        for step in range(self.max_length):
            candidates = []

            for score, seq in beams:
                # Skip completed sequences
                if seq[-1] == end_token:
                    complete_sequences.append((score, seq))
                    continue

                # Prepare input
                tgt = torch.tensor([seq])
                tgt_mask = create_causal_mask(tgt.size(1))

                # Get logits for next token
                logits = self.model.decode(
                    tgt, encoder_output, tgt_mask=tgt_mask
                )

                # Get log probabilities for last position
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

                # Get top k tokens
                topk_log_probs, topk_indices = torch.topk(
                    log_probs[0], beam_size
                )

                # Add candidates
                for log_prob, idx in zip(topk_log_probs, topk_indices):
                    new_score = score + log_prob.item()
                    new_seq = seq + [idx.item()]
                    candidates.append((new_score, new_seq))

            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

            # Early stopping if all beams are complete
            if all(seq[-1] == end_token for _, seq in beams):
                complete_sequences.extend(beams)
                break

        # Add remaining beams
        complete_sequences.extend(beams)

        # Apply length penalty as in paper
        def score_with_penalty(score, length):
            return score / (length ** length_penalty)

        # Sort by length-normalized score
        complete_sequences.sort(
            key=lambda x: score_with_penalty(x[0], len(x[1])),
            reverse=True
        )

        return complete_sequences[0][1] if complete_sequences else beams[0][1]


def demonstrate_generation():
    """
    Demonstrate text generation with the transformer.
    """
    print("="*60)
    print("TRANSFORMER GENERATION DEMO")
    print("="*60)

    # Create model
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        n_heads=8,
        n_layers=6
    )

    # Initialize generator
    generator = TransformerGenerator(model)

    # Dummy source (would be actual tokens in practice)
    src = torch.randint(0, 10000, (1, 10))

    print("\n1. GREEDY GENERATION")
    print("-"*40)
    print("Selecting highest probability token at each step...")
    output_greedy = generator.greedy_generate(
        src, start_token=2, end_token=3
    )
    print(f"Generated: {output_greedy}")

    print("\n2. BEAM SEARCH (Paper's Method)")
    print("-"*40)
    print("Using beam size=4, length penalty=0.6...")
    output_beam = generator.beam_search(
        src, start_token=2, end_token=3,
        beam_size=4, length_penalty=0.6
    )
    print(f"Generated: {output_beam}")

    print("\n" + "="*60)
    print("KEY INSIGHTS FROM THE PAPER:")
    print("="*60)
    print("""
    1. Encoder runs ONCE (efficient!)
    2. Decoder runs autoregressively (one token at a time)
    3. Beam search improves quality over greedy
    4. Length penalty prevents short sequences
    5. This is how real translation systems work!
    """)


if __name__ == "__main__":
    demonstrate_generation()