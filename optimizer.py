"""
Optimizer and Learning Rate Schedule for Transformer
=====================================================
Day 13 of Building Transformers from Scratch

Implements the Adam optimizer with the custom learning rate schedule
from "Attention is All You Need" (2017), Section 5.3.
"""

import torch
import torch.optim as optim
from typing import Optional


class NoamOptimizer:
    """
    Implements learning rate schedule from the paper:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Paper parameters:
    - d_model = 512
    - warmup_steps = 4000
    - Adam: β₁=0.9, β₂=0.98, ε=10⁻⁹
    """

    def __init__(
        self,
        model_size: int,
        warmup_steps: int,
        optimizer: torch.optim.Optimizer
    ):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self._step = 0

    def step(self):
        """Update parameters and learning rate"""
        self._step += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()

    def get_lr(self) -> float:
        """Calculate learning rate based on current step"""
        step = max(1, self._step)
        scale = self.model_size ** (-0.5)

        if step < self.warmup_steps:
            # Linear warmup
            return scale * step * (self.warmup_steps ** (-1.5))
        else:
            # Inverse square root decay
            return scale * (step ** (-0.5))

    def zero_grad(self):
        """Clear gradients"""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Get state for checkpointing"""
        return {
            'step': self._step,
            'optimizer': self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load from checkpoint"""
        self._step = state_dict['step']
        self.optimizer.load_state_dict(state_dict['optimizer'])


def get_std_opt(model: torch.nn.Module, d_model: int = 512) -> NoamOptimizer:
    """
    Get the standard optimizer from Section 5.3:
    "We used Adam optimizer with β₁=0.9, β₂=0.98 and ε=10⁻⁹"

    Args:
        model: The transformer model
        d_model: Model dimension (512 in paper)

    Returns:
        NoamOptimizer with paper's configuration
    """
    # Adam with paper's specific parameters
    # Note: β₂=0.98 differs from PyTorch default (0.999)
    base_optimizer = optim.Adam(
        model.parameters(),
        betas=(0.9, 0.98),  # Paper-specific
        eps=1e-9,  # Paper-specific
        lr=0  # Will be set by NoamOptimizer
    )

    return NoamOptimizer(
        model_size=d_model,
        warmup_steps=4000,  # Paper-specific
        optimizer=base_optimizer
    )


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    PyTorch scheduler version for compatibility with training loops.
    Implements the same schedule as NoamOptimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 512,
        warmup_steps: int = 4000,
        last_epoch: int = -1
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = self.d_model ** (-0.5)

        if step < self.warmup_steps:
            lr = scale * step * (self.warmup_steps ** (-1.5))
        else:
            lr = scale * (step ** (-0.5))

        return [lr for _ in self.base_lrs]