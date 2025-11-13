"""
Training Components for Transformer
====================================
Day 12 of Building Transformers from Scratch

Implements loss functions and training utilities as described in
"Attention is All You Need" (2017), Section 5.4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss from Section 5.4 of the paper.

    The paper uses ε_ls = 0.1, which means:
    - True label gets 0.9 probability
    - Remaining 0.1 is distributed uniformly across all classes

    This prevents the model from becoming overconfident and
    improves generalization (higher BLEU scores).
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,  # [batch_size, seq_len, vocab_size]
        targets: torch.Tensor,  # [batch_size, seq_len]
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Model predictions
            targets: True labels
            mask: Padding mask (1 for valid positions, 0 for padding)

        Returns:
            Scalar loss value
        """
        batch_size, seq_len, _ = logits.shape

        # Reshape for loss calculation
        logits = logits.reshape(-1, self.vocab_size)  # [B*S, V]
        targets = targets.reshape(-1)  # [B*S]

        # Create smoothed target distribution
        # Start with uniform distribution
        smooth_targets = torch.full(
            (targets.size(0), self.vocab_size),
            self.smoothing / (self.vocab_size - 2),  # Exclude pad and true label
            device=logits.device
        )

        # Zero out padding positions
        smooth_targets[:, self.pad_idx] = 0

        # Add confidence to true labels
        smooth_targets.scatter_(
            1, targets.unsqueeze(1), self.confidence
        )

        # For padding tokens, zero out the entire distribution
        pad_mask = targets == self.pad_idx
        smooth_targets[pad_mask] = 0
        smooth_targets[pad_mask, self.pad_idx] = 1  # Padding predicts padding

        # Compute KL divergence (more stable than cross-entropy for smoothed targets)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(
            log_probs,
            smooth_targets,
            reduction='none'
        ).sum(dim=-1)  # Sum over vocab dimension

        # Apply sequence mask if provided
        if mask is not None:
            mask = mask.reshape(-1)
            loss = loss * mask
            return loss.sum() / mask.sum()  # Average over valid positions
        else:
            # Exclude padding from loss
            non_pad_mask = ~pad_mask
            return loss[non_pad_mask].mean()


class TransformerLoss(nn.Module):
    """
    Complete loss function for Transformer training.
    Combines label smoothing with proper masking.
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.criterion = LabelSmoothingLoss(
            vocab_size, smoothing, pad_idx
        )
        self.pad_idx = pad_idx

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss for transformer output.

        Args:
            output: Model output [batch, tgt_len, vocab_size]
            target: Target tokens [batch, tgt_len]

        Returns:
            Scalar loss
        """
        # Shift for next-token prediction
        # Input: [START, tok1, tok2, ...]
        # Target: [tok1, tok2, tok3, ...]
        output = output[:, :-1, :]  # Remove last prediction
        target = target[:, 1:]  # Remove START token

        # Create mask for non-padding positions
        mask = (target != self.pad_idx).float()

        return self.criterion(output, target, mask)


def compute_perplexity(loss: torch.Tensor) -> float:
    """
    Compute perplexity from loss.
    Perplexity = exp(loss)

    Lower perplexity means better language modeling.
    """
    return torch.exp(loss).item()


def demonstrate_label_smoothing():
    """
    Show how label smoothing works with the Transformer.
    """
    print("="*60)
    print("LABEL SMOOTHING DEMONSTRATION")
    print("="*60)

    vocab_size = 100
    batch_size = 2
    seq_len = 5

    # Create sample data
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets[0, -2:] = 0  # Add some padding

    # Without label smoothing
    regular_loss = nn.CrossEntropyLoss(ignore_index=0)
    loss_regular = regular_loss(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )

    # With label smoothing (paper's approach)
    smooth_loss = TransformerLoss(vocab_size, smoothing=0.1, pad_idx=0)
    loss_smooth = smooth_loss(
        torch.cat([torch.zeros(batch_size, 1, vocab_size), logits], dim=1),
        torch.cat([torch.zeros(batch_size, 1).long(), targets], dim=1)
    )

    print(f"\nRegular Cross-Entropy Loss: {loss_regular:.4f}")
    print(f"Label-Smoothed Loss: {loss_smooth:.4f}")
    print(f"\nRegular Perplexity: {compute_perplexity(loss_regular):.2f}")
    print(f"Smoothed Perplexity: {compute_perplexity(loss_smooth):.2f}")

    print("\n" + "="*60)
    print("KEY INSIGHTS FROM THE PAPER:")
    print("="*60)
    print("""
    1. Label smoothing prevents overconfidence
    2. Paper uses ε=0.1 for all experiments
    3. Hurts perplexity but improves BLEU scores
    4. Essential for achieving paper's results
    5. Applied during training, not inference
    """)


if __name__ == "__main__":
    demonstrate_label_smoothing()