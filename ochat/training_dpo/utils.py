"""DPO-specific utilities: batch collation, loss function, and reference log-prob helpers.

Shared utilities (dataset loading, checkpointing, LR scheduling, tokenizer, MLflow)
are in ochat.training_utils._common.
"""

import torch
import torch.nn.functional as F
import numpy as np

from typing import Dict, Optional

from transformers import ProcessorMixin

from ochat.training_utils.numpy_dataset import NumpyDataset
from ochat.training_utils._common import batch_to_tensor


def check_ref_logps_precomputed(train_dataset: NumpyDataset) -> bool:
    """Check whether reference log-probs were precomputed during preprocessing.

    Reads the 'ref_logps_computed' metadata key set by generate_dpo_dataset.
    Falls back to NaN detection for legacy datasets without the metadata key.
    """
    precomputed = train_dataset.metadata.get("ref_logps_computed", None)
    if precomputed is not None:
        return precomputed
    # Legacy dataset: detect via NaN sentinel
    sample = np.asarray(train_dataset["chosen_ref_logp"][0], dtype=np.float32)
    return not np.isnan(sample).any()


def dpo_batch_collate(batch: Dict[str, np.ndarray], dataset_path: Optional[str] = None,
                      processor: Optional[ProcessorMixin] = None):
    """Collate a full DPO batch: both chosen and rejected sides.

    Returns:
        chosen_tensor: dict of tensors for chosen sequences.
        rejected_tensor: dict of tensors for rejected sequences.
        chosen_ref_logps: tensor of pre-computed reference log-probs for chosen.
        rejected_ref_logps: tensor of pre-computed reference log-probs for rejected.
        batch_info: dict with combined max_seqlen and total_seqs.
    """
    chosen_tensor, chosen_info = batch_to_tensor(batch, dataset_path, processor, prefix="chosen_")
    rejected_tensor, rejected_info = batch_to_tensor(batch, dataset_path, processor, prefix="rejected_")

    # Reference log-probs (pre-computed scalars per example)
    # Handle both formats: object arrays (from parquet) and flat arrays (from cache)
    chosen_ref_logps = torch.from_numpy(np.asarray(batch["chosen_ref_logp"], dtype=np.float32).ravel())
    rejected_ref_logps = torch.from_numpy(np.asarray(batch["rejected_ref_logp"], dtype=np.float32).ravel())

    # Combined batch info
    batch_info = {
        "max_seqlen": max(
            chosen_info.get("max_seqlen", 0),
            rejected_info.get("max_seqlen", 0),
        ),
    }

    return chosen_tensor, rejected_tensor, chosen_ref_logps, rejected_ref_logps, batch_info


def dpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard DPO loss.

    L = -log(sigma(beta * ((log_pi_chosen - log_pi_rejected) - (log_ref_chosen - log_ref_rejected))))

    Args:
        chosen_logp: Policy log-prob sum for each chosen response (shape: [B]).
        rejected_logp: Policy log-prob sum for each rejected response (shape: [B]).
        chosen_ref_logp: Reference log-prob sum for each chosen response (shape: [B]).
        rejected_ref_logp: Reference log-prob sum for each rejected response (shape: [B]).
        beta: DPO temperature parameter.
    """
    policy_ratio = chosen_logp - rejected_logp
    ref_ratio = chosen_ref_logp - rejected_ref_logp
    logits = beta * (policy_ratio - ref_ratio)
    return -F.logsigmoid(logits).mean()
