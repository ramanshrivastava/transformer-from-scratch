"""
Byte-Pair Encoding (BPE) Tokenization
======================================
Day 6 of Building Transformers from Scratch

Building on our embedding layer from Day 5, this module shows how
modern LLMs actually create tokens using BPE - the method from the
2017 "Attention is All You Need" paper.

BPE is why ChatGPT never sees an "unknown word" - it can always
fall back to smaller pieces.

"""

import torch
from collections import Counter
from typing import List, Tuple
from embeddings import TransformerEmbedding


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer that learns subword units.
    This feeds directly into our TransformerEmbedding from Day 5.
    """

    def __init__(self):
        # Start with special tokens (same as our SimpleTokenizer)
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[START]": 2,
            "[END]": 3
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = []  # Learned merge rules

    def get_pairs(self, tokens: List[str]) -> Counter:
        """Count frequency of all adjacent token pairs."""
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def merge_pair(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge all occurrences of a token pair."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, texts: List[str], vocab_size: int = 1000) -> None:
        """
        Learn BPE vocabulary from training texts.

        Args:
            texts: Training texts
            vocab_size: Target vocabulary size
        """
        # Collect all text
        all_text = " ".join(texts)
        tokens = list(all_text)

        # Add initial characters to vocab
        for char in set(tokens):
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.id_to_token[idx] = char

        print(f"Starting with {len(self.vocab)} characters")

        # Learn merges until we reach target vocab size
        while len(self.vocab) < vocab_size:
            pairs = self.get_pairs(tokens)
            if not pairs:
                break

            # Get most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            freq = pairs[best_pair]

            # Create new token
            new_token = best_pair[0] + best_pair[1]

            # Add to vocabulary if not present
            if new_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_token] = idx
                self.id_to_token[idx] = new_token

                # Store merge rule
                self.merges.append(best_pair)

                # Apply merge
                tokens = self.merge_pair(tokens, best_pair)

                # Show progress for first few merges
                if len(self.merges) <= 5:
                    print(f"  Merge {len(self.merges)}: '{best_pair[0]}' + '{best_pair[1]}' → '{new_token}' (freq: {freq})")

        print(f"Final vocabulary: {len(self.vocab)} tokens")

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs using learned BPE.

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        # Start with characters
        tokens = list(text)

        # Apply merges in order
        for pair in self.merges:
            tokens = self.merge_pair(tokens, pair)

        # Convert to IDs, use [UNK] for missing tokens
        token_ids = []
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab["[UNK]"]))

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id_to_token.get(tid, "[UNK]") for tid in token_ids]
        return "".join(tokens)  # BPE tokens join without spaces


def demonstrate_bpe_with_embeddings():
    """
    Show how BPE connects to our transformer pipeline from Days 1-5.
    """
    print("="*60)
    print("BPE TOKENIZATION → TRANSFORMER PIPELINE")
    print("="*60)
    print("Connecting Day 6 (BPE) to Days 1-5 (Transformer)\n")

    # Sample training data
    texts = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are animals"
    ]

    # 1. Train BPE tokenizer
    print("STEP 1: Train BPE Tokenizer")
    print("-"*40)
    bpe = BPETokenizer()
    bpe.train(texts, vocab_size=100)

    # 2. Tokenize text
    print("\nSTEP 2: Tokenize Text")
    print("-"*40)
    test_text = "the cat"
    token_ids = bpe.encode(test_text)
    print(f"Text: '{test_text}'")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: '{bpe.decode(token_ids)}'")

    # 3. Feed to our embedding layer (from Day 5)
    print("\nSTEP 3: Feed to TransformerEmbedding (Day 5)")
    print("-"*40)
    embedder = TransformerEmbedding(
        vocab_size=len(bpe.vocab),
        d_model=512
    )

    # Convert to tensor and embed
    token_tensor = torch.tensor([token_ids])
    embedded = embedder(token_tensor)

    print(f"Token tensor shape: {token_tensor.shape}")
    print(f"Embedded shape: {embedded.shape} [batch=1, seq_len={len(token_ids)}, d_model=512]")

    # 4. Ready for transformer
    print("\nSTEP 4: Ready for Transformer Components")
    print("-"*40)
    print("This output can now flow through:")
    print("  → Day 1: Attention mechanism")
    print("  → Day 2: Add positional encoding ✓ (included in embedder)")
    print("  → Day 3: Through transformer block")
    print("  → Day 4: Through 6-layer encoder stack")
    print("  → Day 5: Already embedded ✓")

    return bpe, embedder


def compare_with_word_tokenization():
    """
    Show why BPE beats word-based tokenization.
    """
    print("\n" + "="*60)
    print("WHY BPE > WORD TOKENIZATION")
    print("="*60)

    # Train BPE on simple corpus
    bpe = BPETokenizer()
    bpe.train(["the cat sat on the mat"], vocab_size=50)

    test_cases = [
        ("Known words", "the cat"),
        ("Unknown word", "supercalifragilisticexpialidocious"),
        ("New tech term", "ChatGPT"),
        ("Mixed", "the ChatGPT cat"),
    ]

    for label, text in test_cases:
        print(f"\n{label}: '{text}'")
        print("-"*40)

        # Word tokenization (simulation)
        print("  Word tokenization:")
        if text in ["the cat"]:
            print(f"    → {text.split()}")
        else:
            print(f"    → Contains [UNKNOWN] tokens")

        # BPE tokenization
        tokens = bpe.encode(text)
        decoded = bpe.decode(tokens)
        print("  BPE tokenization:")
        print(f"    → Token IDs: {tokens}")
        print(f"    → Decoded: '{decoded}'")
        print(f"    → Never fails! Falls back to characters")


if __name__ == "__main__":
    # Show complete pipeline
    bpe, embedder = demonstrate_bpe_with_embeddings()

    # Show comparison
    compare_with_word_tokenization()

    print("\n" + "="*60)
    print("THE COMPLETE PIPELINE (Days 1-6)")
    print("="*60)
    print("""
    Text ("the cat sat")
         ↓
    BPE Tokenization (Day 6) → [4, 87, 23]
         ↓
    Embeddings (Day 5) → [[0.1, -0.3, ...], ...]
         ↓
    + Positional Encoding (Day 2)
         ↓
    Multi-Head Attention (Day 1)
         ↓
    Transformer Block (Day 3)
         ↓
    6-Layer Encoder Stack (Day 4)
         ↓
    Understanding!

    Next: Decoder for generation (Day 7)
    """)