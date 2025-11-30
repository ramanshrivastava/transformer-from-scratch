"""
Model Checkpointing for Transformer
====================================
Day 15 of Building Transformers from Scratch

Save and load model state during training.

From the paper (Section 6.1):
"For base models, we averaged the last 5 checkpoints"
"For big models, we averaged the last 20 checkpoints"

Checkpointing serves TWO purposes:
1. Resume training after crashes
2. Average checkpoints for better inference
"""

import torch
from typing import List, Tuple


def save_checkpoint(
    model,
    optimizer,
    step: int,
    loss: float,
    path: str
):
    """
    Save everything needed to resume training.

    What we save:
    - model weights (the learned parameters)
    - optimizer state (momentum, learning rate step)
    - step number (where we left off)
    - loss value (to track progress)
    """
    checkpoint = {
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: step {step}, loss {loss:.4f}")


def load_checkpoint(
    model,
    optimizer,
    path: str
) -> Tuple[int, float]:
    """
    Load a saved checkpoint to resume training.

    Returns:
        step: Where to resume from
        loss: Last recorded loss
    """
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from step {checkpoint['step']}")
    return checkpoint['step'], checkpoint['loss']


def save_best_model(model, path: str):
    """
    Save just the model weights (for inference).
    Smaller file, only what's needed to generate.
    """
    torch.save(model.state_dict(), path)
    print(f"Saved best model to {path}")


def load_model_for_inference(model, path: str):
    """
    Load model weights for generation (no optimizer needed).
    """
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def average_checkpoints(checkpoint_paths: List[str], model):
    """
    Average multiple checkpoints for better inference.

    From paper Section 6.1:
    "For base models, we averaged the last 5 checkpoints"
    "For big models, we averaged the last 20 checkpoints"

    Why this works:
    - Each checkpoint captures slightly different optima
    - Averaging smooths out training noise
    - Results in more robust predictions
    - Free performance boost (+0.5 BLEU in some cases)

    Args:
        checkpoint_paths: List of checkpoint file paths
        model: Model to load averaged weights into

    Returns:
        Model with averaged weights, in eval mode
    """
    print(f"Averaging {len(checkpoint_paths)} checkpoints...")

    # Load all checkpoint state dicts
    state_dicts = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path)
        state_dicts.append(checkpoint['model_state_dict'])

    # Average the weights
    avg_state = {}
    for key in state_dicts[0].keys():
        # Convert to float for averaging, then back to original dtype
        tensors = [sd[key].float() for sd in state_dicts]
        avg_state[key] = torch.stack(tensors).mean(dim=0)

        # Restore original dtype
        avg_state[key] = avg_state[key].to(state_dicts[0][key].dtype)

    # Load averaged weights into model
    model.load_state_dict(avg_state)
    model.eval()

    print("Checkpoint averaging complete!")
    return model
