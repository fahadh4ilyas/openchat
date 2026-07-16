"""ORPO utilities: loss function, batch collation, and shared helpers.

ORPO (Odds Ratio Preference Optimization) combines SFT and preference
alignment in a single objective without a reference model.

Reference: Hong et al., "ORPO: Monolithic Preference Optimization without
Reference Model" (arXiv:2403.07691).
"""

import os
import math
import json
import shutil
import mlflow
import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

from typing import Dict, Optional
from pathlib import Path
from functools import partial

from transformers import ProcessorMixin

from ochat.config import MODEL_CONFIG_MAP
from ochat.training_utils.numpy_dataset import NumpyDataset
from ochat.training_dpo.utils import (
    batch_to_tensor,
    _BATCH_KEYS,
    create_dataset,
    calculate_auto_lr,
    create_lr_scheduler,
    save_tokenizer,
    load_tokenizer,
    save_openchat_metadata,
    clean_checkpoint,
    get_latest_checkpoint,
    mlflow_stopper_wrapper,
)


PAD_ID = 0


def log1mexp(x: torch.FloatTensor) -> torch.FloatTensor:
    """Numerically stable computation of log(1 - exp(x)) for x <= 0.

    Branches at -ln 2 ≈ -0.693 to avoid catastrophic cancellation.
    Source: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    t = -0.6931471805599453
    return torch.where(x < t, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


def orpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """ORPO odds-ratio loss (per-example, averaged over batch).

    Implements Eq. (4) and (7) from Hong et al. (2024):
        L_ORPO = -log σ(β · log(odds_chosen / odds_rejected))

    where log(odds) = log_p - log(1 - exp(log_p)) = log_p - log1mexp(log_p).

    Args:
        chosen_logp: Policy log-prob sum for each chosen response (shape: [B]).
        rejected_logp: Policy log-prob sum for each rejected response (shape: [B]).
        beta: Temperature / weight parameter (λ in the paper, default 0.1).

    Returns:
        Scalar loss = -mean(log σ(β · Δ_log_odds)).
    """
    chosen_logp = chosen_logp.float()
    rejected_logp = rejected_logp.float()

    log_odds_chosen = chosen_logp - log1mexp(chosen_logp)
    log_odds_rejected = rejected_logp - log1mexp(rejected_logp)
    log_odds_ratio = log_odds_chosen - log_odds_rejected

    return -F.logsigmoid(beta * log_odds_ratio).mean()


def orpo_batch_collate(batch: Dict[str, np.ndarray], dataset_path: Optional[str] = None,
                       processor: Optional[ProcessorMixin] = None):
    """Collate a batch for ORPO: chosen + rejected tensors (no reference log-probs).

    ORPO uses the same paired data format as DPO but ignores ref_logp fields.
    """
    chosen_tensor = batch_to_tensor(batch, dataset_path, processor, prefix="chosen_")
    rejected_tensor = batch_to_tensor(batch, dataset_path, processor, prefix="rejected_")

    batch_info = {
        "max_seqlen": max(
            chosen_tensor.get("max_seqlen", 0),
            rejected_tensor.get("max_seqlen", 0),
        ),
    }

    return chosen_tensor, rejected_tensor, batch_info
