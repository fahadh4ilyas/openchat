"""KTO-specific utilities: batch collation, loss function, and reference log-prob helpers.

KTO (Kahneman-Tversky Optimization) uses unpaired preference data where each
example is independently labeled as desirable or undesirable. No chosen/rejected
pairing — just a single forward pass per batch.

Reference: Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic
Optimization" (arXiv:2402.01306).
"""

import torch
import torch.nn.functional as F
import numpy as np

from typing import Dict, Optional

from transformers import ProcessorMixin

from ochat.training_utils.numpy_dataset import NumpyDataset
from ochat.training_utils._common import batch_to_tensor


def check_ref_logps_precomputed(train_dataset: NumpyDataset) -> bool:
    """Check whether KTO reference log-probs were precomputed during preprocessing.

    Reads the 'ref_logps_computed' metadata key set by generate_dataset --kto.
    Falls back to NaN detection for legacy datasets without the metadata key.
    """
    precomputed = train_dataset.metadata.get("ref_logps_computed", None)
    if precomputed is not None:
        return precomputed
    sample = np.asarray(train_dataset["ref_logp"][0], dtype=np.float32)
    return not np.isnan(sample).any()


def kto_batch_collate(batch: Dict[str, np.ndarray], dataset_path: Optional[str] = None,
                      processor: Optional[ProcessorMixin] = None):
    """Collate a KTO batch: SFT-style tensors + labels + reference log-probs.

    Returns:
        tensor: dict of tensors for the batch (same as SFT batch_to_tensor).
        batch_info: dict with max_seqlen.
        labels: bool tensor of shape (B,) — True = desirable, False = undesirable.
        ref_logps: float tensor of shape (B,) — pre-computed reference log-probs.
    """
    tensor, batch_info = batch_to_tensor(batch, dataset_path, processor)

    labels = torch.from_numpy(np.asarray(batch["label"], dtype=bool).ravel())
    ref_logps = torch.from_numpy(np.asarray(batch["ref_logp"], dtype=np.float32).ravel())

    return tensor, batch_info, labels, ref_logps


def kto_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """KTO loss for unpaired preference data.

    For each example:
        log_ratio = log_p - log_p_ref
        kl = mean(log_ratio) over the batch (batch-level KL estimator)

        desirable_loss   = 1 - sigmoid(beta * (log_ratio - kl))
        undesirable_loss  = 1 - sigmoid(beta * (kl - log_ratio))

    Args:
        policy_logps: Policy log-prob sum for each response (shape: [B]).
        ref_logps: Reference log-prob sum for each response (shape: [B]).
        labels: bool tensor, True = desirable, False = undesirable (shape: [B]).
        beta: KTO temperature parameter (default 0.1).

    Returns:
        Scalar loss averaged over the batch.
    """
    log_ratios = policy_logps.float() - ref_logps.float()
    kl = log_ratios.mean()

    desirable_mask = labels.to(torch.bool)
    undesirable_mask = ~desirable_mask

    losses = torch.empty_like(log_ratios)

    if desirable_mask.any():
        losses[desirable_mask] = 1 - torch.sigmoid(beta * (log_ratios[desirable_mask] - kl))

    if undesirable_mask.any():
        losses[undesirable_mask] = 1 - torch.sigmoid(beta * (kl - log_ratios[undesirable_mask]))

    return losses.mean()
