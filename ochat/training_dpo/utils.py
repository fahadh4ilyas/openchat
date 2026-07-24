"""DPO-specific utilities: batch collation, loss functions, and reference log-prob helpers.

Shared utilities (dataset loading, checkpointing, LR scheduling, tokenizer, MLflow)
are in ochat.training_utils._common.

Loss variants adapted from TRL's DPOTrainer (https://github.com/huggingface/trl).
"""

import torch
import torch.nn.functional as F
import numpy as np

from typing import Dict, Optional, Tuple

from transformers import ProcessorMixin

from ochat.training_utils._common import batch_to_tensor


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


# -- Core log-ratio helpers ----------------------------------------------------


def _compute_logratios(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute chosen/rejected log-ratios and their difference.

    Returns:
        chosen_logratios, rejected_logratios, delta_score
    """
    chosen_logratios = chosen_logp - chosen_ref_logp
    rejected_logratios = rejected_logp - rejected_ref_logp
    delta_score = chosen_logratios - rejected_logratios
    return chosen_logratios, rejected_logratios, delta_score


# -- Standard DPO loss ---------------------------------------------------------


def dpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Standard DPO loss (sigmoid).

    L = -log(sigma(beta * ((log_pi_chosen - log_pi_rejected) - (log_ref_chosen - log_ref_rejected))))
    """
    _, _, delta = _compute_logratios(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp)
    return -F.logsigmoid(beta * delta).mean()


# -- Loss variants -------------------------------------------------------------


def hinge_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Hinge loss: max(0, 1 - beta * delta)."""
    _, _, delta = _compute_logratios(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp)
    return torch.relu(1 - beta * delta).mean()


def ipo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
    chosen_tokens: torch.Tensor,
    rejected_tokens: torch.Tensor,
) -> torch.Tensor:
    """IPO loss: (delta_avg - 1/(2*beta))^2 using per-token average log-probs.

    beta here is τ (the regularization parameter in the IPO paper).
    """
    chosen_logratios, rejected_logratios, _ = _compute_logratios(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp
    )
    chosen_avg = chosen_logratios / chosen_tokens.clamp(min=1.0)
    rejected_avg = rejected_logratios / rejected_tokens.clamp(min=1.0)
    ipo_delta = chosen_avg - rejected_avg
    return (ipo_delta - 1 / (2 * beta)) ** 2


def exo_pair_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """EXO-pref loss: KL(p_theta || p_rh) for K=2.

    p_rh = [(1-ε), ε]; expanded KL gives weighted logsigmoid form.
    Reference: https://huggingface.co/papers/2402.00856, Eq. 16.
    """
    _, _, delta = _compute_logratios(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp)
    eps = label_smoothing
    qw = torch.sigmoid(beta * delta)
    log_qw = F.logsigmoid(beta * delta)
    log_pw = torch.log1p(-eps)
    ql = torch.sigmoid(-beta * delta)
    log_ql = F.logsigmoid(-beta * delta)
    log_pl = torch.log(eps)
    return (qw * (log_qw - log_pw) + ql * (log_ql - log_pl)).mean()


def nca_pair_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """NCA loss: -log_sigma(r_chosen) - 0.5*log_sigma(-r_chosen) - 0.5*log_sigma(-r_rejected)."""
    chosen_logratios, rejected_logratios, _ = _compute_logratios(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp
    )
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios
    return (
        -F.logsigmoid(chosen_rewards)
        - 0.5 * F.logsigmoid(-chosen_rewards)
        - 0.5 * F.logsigmoid(-rejected_rewards)
    ).mean()


def robust_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Robust DPO loss: models ε label-flip probability.

    L = -(1-ε)*log_sigma(beta*delta) - ε*log_sigma(-beta*delta).
    """
    _, _, delta = _compute_logratios(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp)
    eps = label_smoothing
    return (
        -(1 - eps) * F.logsigmoid(beta * delta)
        - eps * F.logsigmoid(-beta * delta)
    ).mean()


def bco_pair_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """BCO loss: -log_sigma(r_chosen) - log_sigma(-r_rejected)."""
    chosen_logratios, rejected_logratios, _ = _compute_logratios(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp
    )
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios
    return (-F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)).mean()


def sppo_hard_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """SPPO hard loss: -beta * delta (no log-sigmoid)."""
    _, _, delta = _compute_logratios(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp)
    return (-beta * delta).mean()


def aot_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """AOT paired loss: sort by score, apply robust loss to sorted pairs.

    Alignment Over Trajectories — sorts chosen/rejected jointly by their
    policy log-ratios and computes a robust DPO-style loss over sorted pairs.
    """
    chosen_logratios, rejected_logratios, _ = _compute_logratios(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp
    )
    chosen_scores = beta * chosen_logratios
    rejected_scores = beta * rejected_logratios
    # Sort chosen ascending, rejected descending, then pair
    chosen_sorted, _ = torch.sort(chosen_scores)
    rejected_sorted, _ = torch.sort(rejected_scores, descending=True)
    delta = chosen_sorted - rejected_sorted
    eps = label_smoothing
    return (
        -F.logsigmoid(beta * delta) * (1 - eps)
        - F.logsigmoid(-beta * delta) * eps
    ).mean()


def aot_unpaired_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """AOT unpaired loss: robust loss over all score deltas (not sorted pairs)."""
    chosen_logratios, rejected_logratios, _ = _compute_logratios(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp
    )
    chosen_scores = beta * chosen_logratios
    rejected_scores = beta * rejected_logratios
    # All pairwise deltas
    delta = chosen_scores.unsqueeze(1) - rejected_scores.unsqueeze(0)
    eps = label_smoothing
    losses = (
        -F.logsigmoid(beta * delta) * (1 - eps)
        - F.logsigmoid(-beta * delta) * eps
    )
    return losses.mean()


def apo_zero_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """APO-zero loss: (1 - sigma(beta*lr_chosen)) + sigma(beta*lr_rejected).

    Reference: https://huggingface.co/papers/2408.06266, Eq. 7.
    """
    chosen_logratios, rejected_logratios, _ = _compute_logratios(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp
    )
    losses_chosen = 1 - torch.sigmoid(beta * chosen_logratios)
    losses_rejected = torch.sigmoid(beta * rejected_logratios)
    return torch.cat([losses_chosen, losses_rejected]).mean()


def apo_down_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """APO-down loss: sigma(beta*lr_chosen) + (1 - sigma(beta*delta)).

    Reference: https://huggingface.co/papers/2408.06266, Eq. 8.
    """
    chosen_logratios, rejected_logratios, _ = _compute_logratios(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp
    )
    delta = chosen_logratios - rejected_logratios
    losses_chosen = torch.sigmoid(beta * chosen_logratios)
    losses_rejected = 1 - torch.sigmoid(beta * delta)
    return torch.cat([losses_chosen, losses_rejected]).mean()


def discopop_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
    discopop_tau: float = 0.05,
) -> torch.Tensor:
    """DiscoPOP loss: sigmoid-gated modulation of standard DPO loss.

    Reference: https://huggingface.co/papers/2406.08414, Eq. 5.
    """
    _, _, delta = _compute_logratios(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp)
    logits = beta * delta
    log_ratio_modulation = torch.sigmoid(logits / discopop_tau)
    return (log_ratio_modulation * (-F.logsigmoid(logits))).mean()


def sft_loss(
    chosen_logp: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Pure SFT loss on chosen responses: -log p_theta(chosen)."""
    return -chosen_logp.mean()


def sigmoid_norm_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
    chosen_tokens: torch.Tensor,
    rejected_tokens: torch.Tensor,
) -> torch.Tensor:
    """Sigmoid loss with length-normalized delta."""
    chosen_logratios, rejected_logratios, _ = _compute_logratios(
        chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp
    )
    total_tokens = (chosen_tokens + rejected_tokens).clamp(min=1.0)
    delta_norm = (chosen_logratios - rejected_logratios) / total_tokens
    return -F.logsigmoid(beta * delta_norm).mean()


# -- Loss router ---------------------------------------------------------------


# Loss variants that only need log-probs + beta (no extra args)
_SIMPLE_VARIANTS = {
    "sigmoid": dpo_loss,
    "hinge": hinge_loss,
    "nca_pair": nca_pair_loss,
    "bco_pair": bco_pair_loss,
    "sppo_hard": sppo_hard_loss,
    "apo_zero": apo_zero_loss,
    "apo_down": apo_down_loss,
}

# Loss variants that need token counts
_TOKEN_AWARE_VARIANTS = {
    "ipo": ipo_loss,
    "sigmoid_norm": sigmoid_norm_loss,
}

# Loss variants that need label_smoothing
_SMOOTHING_VARIANTS = {
    "exo_pair": exo_pair_loss,
    "robust": robust_loss,
    "aot": aot_loss,
    "aot_unpaired": aot_unpaired_loss,
}

# Loss variants with special requirements
_SPECIAL_VARIANTS = {
    "discopop": discopop_loss,
    "sft": sft_loss,
}


_VALID_LOSS_TYPES = set(
    list(_SIMPLE_VARIANTS) + list(_TOKEN_AWARE_VARIANTS)
    + list(_SMOOTHING_VARIANTS) + list(_SPECIAL_VARIANTS)
)


def dpo_loss_router(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
    loss_type: str = "sigmoid",
    chosen_tokens: torch.Tensor | None = None,
    rejected_tokens: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    discopop_tau: float = 0.05,
) -> torch.Tensor:
    """Route to the appropriate DPO loss variant.

    Args:
        chosen_logp: Policy log-prob sum for each chosen response (shape: [B]).
        rejected_logp: Policy log-prob sum for each rejected response (shape: [B]).
        chosen_ref_logp: Reference log-prob sum for each chosen response (shape: [B]).
        rejected_ref_logp: Reference log-prob sum for each rejected response (shape: [B]).
        beta: DPO temperature parameter.
        loss_type: One of the 15 supported loss types.
        chosen_tokens: Per-sequence response token counts for chosen (required for ipo, sigmoid_norm).
        rejected_tokens: Per-sequence response token counts for rejected (required for ipo, sigmoid_norm).
        label_smoothing: Label smoothing ε (used by exo_pair, robust, aot, aot_unpaired).
        discopop_tau: Temperature τ for DiscoPOP modulation.

    Returns:
        Scalar loss tensor.
    """
    if loss_type not in _VALID_LOSS_TYPES:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. Valid types: {sorted(_VALID_LOSS_TYPES)}"
        )

    if loss_type in _SIMPLE_VARIANTS:
        return _SIMPLE_VARIANTS[loss_type](chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta)

    if loss_type in _TOKEN_AWARE_VARIANTS:
        if chosen_tokens is None or rejected_tokens is None:
            raise ValueError(f"loss_type='{loss_type}' requires chosen_tokens and rejected_tokens")
        return _TOKEN_AWARE_VARIANTS[loss_type](
            chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta,
            chosen_tokens, rejected_tokens,
        )

    if loss_type in _SMOOTHING_VARIANTS:
        return _SMOOTHING_VARIANTS[loss_type](
            chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta,
            label_smoothing=label_smoothing,
        )

    if loss_type == "discopop":
        return discopop_loss(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta, discopop_tau)

    if loss_type == "sft":
        return sft_loss(chosen_logp)


# -- CPO loss (built on DPO loss) ----------------------------------------------


def cpo_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_ref_logp: torch.Tensor,
    rejected_ref_logp: torch.Tensor,
    beta: float,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPO (Contrastive Preference Optimization) loss.

    L_CPO = L_DPO + α · L_SFT
    where L_SFT = -log p_θ(chosen) = -chosen_logp

    Reference: Xu et al., "Contrastive Preference Optimization: Pushing the
    Boundaries of LLM Performance in Machine Translation" (arXiv:2401.08417).

    Returns:
        (total_loss, dpo_loss_component, sft_loss_component)
    """
    dpo = dpo_loss(chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp, beta)
    sft = -chosen_logp.mean()
    return dpo + alpha * sft, dpo, sft
