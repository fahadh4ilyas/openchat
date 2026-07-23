"""Shared cross-entropy and loss utilities for unpadded model forward methods.

Consolidates duplicated code from 25+ model files:
- weighted_cross_entropy (was copy-pasted across all models)
- weighted_token_accuracy (was copy-pasted across all models)
- compute_unpadded_lm_loss (the shared forward loss pattern)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

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


@torch.compile
def weighted_token_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
):
    """Weighted token prediction accuracy.

    Args:
        logits: (N, V) logits.
        labels: (N,) label indices.
        weights: (N,) per-token weights.

    Returns:
        Scalar sum of weights for correctly predicted tokens.
    """
    return (weights * (torch.argmax(logits, dim=-1) == labels)).sum()


def compute_unpadded_lm_loss(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    nz_shifted_label_ids: torch.Tensor,
    nz_shifted_loss_weights: torch.Tensor,
    lm_head: nn.Module,
    num_tokens: int = 0,
    return_per_seq_logps: bool = False,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
    """Compute CE loss and optionally per-sequence log-prob sums.

    This is the shared loss pattern extracted from all unpadded model forward
    methods (previously copy-pasted across 25+ model files).

    Args:
        hidden_states: (total_tokens, hidden_dim) from the backbone.
        cu_seqlens: Cumulative sequence lengths, shape (num_seqs + 1,).
        nz_shifted_label_ids: (total_tokens,) labels (shifted by 1).
        nz_shifted_loss_weights: (total_tokens,) per-token loss weights.
        lm_head: The lm_head Linear layer for vocab projection.
        num_tokens: Normalization denominator. Pass ``num_seq`` (standard) or
            ``total_seqs`` (ring).  If 0, returns raw totals (used by DPO/ORPO).
        return_per_seq_logps: If True, also compute per-sequence log-prob sums
            (used by DPO/ORPO for splitting chosen/rejected after concatenation).

    Returns:
        loss: Tuple of (ce_loss, accuracy) — both scalars.
        per_seq_logps: (num_seqs,) tensor of per-seq log-prob sums, or None.
    """
    logits = lm_head(hidden_states)

    per_seq_logps = None
    if return_per_seq_logps:
        # DPO/ORPO path: only per_seq_logps is used by callers.
        # Compute per-token CE losses and aggregate per sequence.
        # Skip total_loss/total_acc — they would be discarded anyway.
        num_seqs = int(cu_seqlens.shape[0] - 1)
        token_losses = weighted_cross_entropy(
            logits, nz_shifted_label_ids, nz_shifted_loss_weights, reduction="none"
        )
        positions = torch.arange(logits.size(0), device=logits.device)
        seq_indices = torch.searchsorted(cu_seqlens, positions, right=True) - 1
        per_seq_loss = torch.zeros(
            num_seqs, device=logits.device, dtype=token_losses.dtype
        )
        per_seq_loss.index_add_(0, seq_indices, token_losses)
        per_seq_logps = -per_seq_loss

        # Return dummy loss tuple — callers (DPO/ORPO) never read it.
        loss = (token_losses.sum(), torch.tensor(0.0, device=logits.device))
        return loss, per_seq_logps

    # SFT path: compute total CE loss and token accuracy.
    total_loss = weighted_cross_entropy(
        logits, nz_shifted_label_ids, nz_shifted_loss_weights
    )
    total_acc = weighted_token_accuracy(
        logits.detach(), nz_shifted_label_ids, nz_shifted_loss_weights
    )

    if num_tokens > 0:
        return (total_loss / num_tokens, total_acc / num_tokens), per_seq_logps
    else:
        return (total_loss, total_acc), per_seq_logps
