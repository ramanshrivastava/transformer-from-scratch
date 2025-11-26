"""
Training Loop for Transformer
==============================
Day 14 of Building Transformers from Scratch

The 4 steps every neural network uses to learn:
1. Forward  - Make a prediction
2. Loss     - Measure how wrong it was
3. Backward - Find what caused the error
4. Update   - Adjust weights to improve
"""

import torch
import torch.nn as nn


def train_step(model, optimizer, criterion, src, tgt):
    """
    One step of training. This is where learning happens!

    Args:
        model: Our Transformer
        optimizer: Noam optimizer (handles learning rate)
        criterion: Label smoothing loss
        src: Source sentence tokens [batch, src_len]
        tgt: Target sentence tokens [batch, tgt_len]

    Returns:
        loss_value: How wrong the prediction was (lower = better)
    """
    model.train()  # Enable dropout

    # ═══════════════════════════════════════════════════════
    # STEP 1: FORWARD - Make a prediction
    # ═══════════════════════════════════════════════════════
    # Teacher forcing: give decoder the real answer tokens
    # Input:  [START, I, love, you]
    # We give: [START, I, love] → model predicts → [I, love, you]

    decoder_input = tgt[:, :-1]  # Everything except last token
    prediction = model(src, decoder_input)

    # ═══════════════════════════════════════════════════════
    # STEP 2: LOSS - How wrong was it?
    # ═══════════════════════════════════════════════════════
    # Compare prediction to actual target
    # Target: [I, love, you, END] (shifted by 1)

    target = tgt[:, 1:]  # Everything except first token
    loss = criterion(prediction, target)

    # ═══════════════════════════════════════════════════════
    # STEP 3: BACKWARD - What caused the mistake?
    # ═══════════════════════════════════════════════════════
    # Calculate gradients: how much each weight contributed to error

    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Calculate new gradients

    # Prevent exploding gradients (keep training stable)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # ═══════════════════════════════════════════════════════
    # STEP 4: UPDATE - Adjust to do better next time
    # ═══════════════════════════════════════════════════════
    # Nudge weights in direction that reduces error

    optimizer.step()  # Update weights using gradients

    return loss.item()  # Return loss value for tracking


def evaluate(model, criterion, src, tgt):
    """
    Test how well the model does WITHOUT updating weights.
    Used to check if model generalizes to new data.
    """
    model.eval()  # Disable dropout

    with torch.no_grad():  # Don't calculate gradients (faster)
        decoder_input = tgt[:, :-1]
        prediction = model(src, decoder_input)
        target = tgt[:, 1:]
        loss = criterion(prediction, target)

    return loss.item()


def train_epoch(model, optimizer, criterion, data_batches):
    """
    Train on all batches once (one epoch).

    Returns:
        average_loss: Mean loss across all batches
    """
    total_loss = 0
    num_batches = 0

    for src_batch, tgt_batch in data_batches:
        loss = train_step(model, optimizer, criterion, src_batch, tgt_batch)
        total_loss += loss
        num_batches += 1

        # Print progress every 100 batches
        if num_batches % 100 == 0:
            print(f"  Batch {num_batches}: loss = {loss:.4f}")

    return total_loss / num_batches
