"""
Token Embeddings from 'Attention is All You Need' (2017)
=========================================================
This module implements the input pipeline that converts text to vectors:
1. Tokenization: Text → Token IDs
2. Embedding: Token IDs → Vectors
3. Positioning: Add positional information

This is how transformers "read" text.

Author: Building Transformers from Scratch Series
Date: 2025
"""

import torch
import torch.nn as nn
import math
from typing import List, Dict
from attention_mechanism import PositionalEncoding


class SimpleTokenizer:
    """
    Simple word-level tokenizer for demonstration.
    The paper used Byte-Pair Encoding with ~37K tokens,
    but we use word-level for clarity.
    """

    def __init__(self):
        # Special tokens
        self.vocab: Dict[str, int] = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[START]": 2,
            "[END]": 3
        }
        self.id_to_token: Dict[int, str] = {
            0: "[PAD]",
            1: "[UNK]",
            2: "[START]",
            3: "[END]"
        }

    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000):
        """
        Build vocabulary from training texts.

        Args:
            texts: List of text strings
            max_vocab_size: Maximum vocabulary size
        """
        word_freq = {}

        # Count word frequencies
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and add to vocab
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        for word, _ in sorted_words[:max_vocab_size - len(self.vocab)]:
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.id_to_token[idx] = word

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        return [self.vocab.get(word.lower(), self.vocab["[UNK]"])
                for word in text.split()]

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        return " ".join([self.id_to_token.get(tid, "[UNK]")
                        for tid in token_ids])


class TransformerEmbedding(nn.Module):
    """
    Embedding layer exactly as described in 'Attention is All You Need'.
    Combines learned token embeddings with positional encoding.

    The key insight: multiply embeddings by sqrt(d_model) to match
    the scale of positional encodings.
    """

    def __init__(self, vocab_size: int, d_model: int = 512,
                 max_seq_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of embeddings (512 in paper)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        # Learned embedding matrix [vocab_size, d_model]
        # Each token gets a d_model-dimensional vector
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Scaling factor - critical for proper scaling!
        # Without this, embeddings are too small compared to positions
        self.scale = math.sqrt(d_model)

        # Reuse the positional encoding from our previous work
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Initialize embeddings (normal distribution as in the paper)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=d_model**-0.5)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to positioned embeddings.

        Args:
            token_ids: Tensor of token indices [batch_size, seq_len]

        Returns:
            Embedded and positioned vectors [batch_size, seq_len, d_model]
        """
        # Step 1: Look up embeddings for each token ID
        # This is just indexing into the embedding matrix
        embedded = self.token_embedding(token_ids)

        # Step 2: Scale by sqrt(d_model)
        # This matches the magnitude of positional encodings
        scaled = embedded * self.scale

        # Step 3: Add positional encoding and apply dropout
        output = self.pos_encoding(scaled)

        return output


def demonstrate_embedding_pipeline():
    """
    Complete demonstration of text → tokens → embeddings → transformer input.
    """
    print("="*60)
    print("TEXT TO TRANSFORMER: THE COMPLETE PIPELINE")
    print("="*60)

    # Sample texts for vocabulary building
    training_texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "transformers revolutionized natural language processing"
    ]

    # Step 1: Build tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(training_texts)
    print(f"\n1. TOKENIZATION")
    print(f"   Vocabulary size: {len(tokenizer.vocab)}")
    print(f"   Sample vocab: {dict(list(tokenizer.vocab.items())[:10])}")

    # Step 2: Tokenize text
    text = "the cat sat"
    token_ids = tokenizer.encode(text)
    print(f"\n2. TEXT TO TOKENS")
    print(f"   Input: '{text}'")
    print(f"   Token IDs: {token_ids}")
    print(f"   Decoded: '{tokenizer.decode(token_ids)}'")

    # Step 3: Create embedder
    vocab_size = len(tokenizer.vocab)
    d_model = 512
    embedder = TransformerEmbedding(vocab_size, d_model)
    embedder.eval()

    # Step 4: Embed tokens
    token_tensor = torch.tensor([token_ids])  # Add batch dimension
    with torch.no_grad():
        embedded = embedder(token_tensor)

    print(f"\n3. TOKENS TO EMBEDDINGS")
    print(f"   Token tensor shape: {token_tensor.shape}")
    print(f"   Embedded shape: {embedded.shape}")
    print(f"   Embedding dimensions: [batch_size=1, seq_len={len(token_ids)}, d_model={d_model}]")

    # Show some statistics
    print(f"\n4. EMBEDDING STATISTICS")
    print(f"   Mean: {embedded.mean().item():.4f}")
    print(f"   Std: {embedded.std().item():.4f}")
    print(f"   Min: {embedded.min().item():.4f}")
    print(f"   Max: {embedded.max().item():.4f}")

    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("""
1. TOKENIZATION IS JUST A DICTIONARY:
   - Each unique word gets a unique ID
   - "the" → 4, "cat" → 5, "sat" → 6, etc.
   - No learning involved, just lookup

2. EMBEDDINGS START RANDOM:
   - Each token ID maps to 512 random numbers initially
   - Through training, similar words get similar vectors
   - "cat" and "dog" vectors become similar, "cat" and "airplane" different

3. THE √512 SCALING:
   - Embeddings are multiplied by sqrt(d_model) ≈ 22.6
   - This balances them with positional encodings
   - Without this, positions might dominate or be ignored

4. READY FOR TRANSFORMER:
   - Output shape [batch, seq_len, d_model] feeds directly to encoder
   - This same pipeline is used by GPT, BERT, and all modern LLMs
   - The foundation of the entire NLP revolution
    """)

    return tokenizer, embedder


if __name__ == "__main__":
    # Run demonstration
    tokenizer, embedder = demonstrate_embedding_pipeline()

    # Example: Process multiple sentences
    print("\n" + "="*60)
    print("PROCESSING MULTIPLE SENTENCES")
    print("="*60)

    sentences = [
        "transformers are powerful",
        "the cat sat",
        "natural language processing"
    ]

    for sentence in sentences:
        tokens = tokenizer.encode(sentence)
        token_tensor = torch.tensor([tokens])

        with torch.no_grad():
            embedded = embedder(token_tensor)

        print(f"\n'{sentence}'")
        print(f"  Tokens: {tokens}")
        print(f"  Embedded shape: {embedded.shape}")