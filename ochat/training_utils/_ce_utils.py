"""Shared cross-entropy utilities for unpadded model forward methods.

Replaces duplicated `weighted_cross_entropy` functions across model files.
"""

import torch

from flash_attn.ops.triton.cross_entropy import cross_entropy_loss


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    reduction: str = "sum",
):
    """Weighted cross-entropy loss.

    Args:
        logits: (N, V) logits.
        labels: (N,) label indices.
        weights: (N,) per-token weights.
        reduction: "sum" → scalar total, "none" → per-token (N,).

    Returns:
        If reduction="sum": scalar loss.
        If reduction="none": (N,) per-token losses.
    """
    loss_per_token = weights * cross_entropy_loss(logits, labels, inplace_backward=True)[0]
    if reduction == "sum":
        return loss_per_token.sum()
    return loss_per_token
