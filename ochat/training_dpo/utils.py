"""DPO utilities: batch collation, loss function, and shared helpers."""

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


PAD_ID = 0

# Batch keys for chosen and rejected sequences (same as standard BATCH_KEYS)
_BATCH_KEYS = {
    "seqlens": torch.long,
    "nz_input_ids": torch.long,
    "nz_position_ids": torch.long,
    "nz_shifted_label_ids": torch.long,
    "nz_shifted_loss_weights": torch.bfloat16,
}

MODEL_LR = ["mistral", "mixtral", "qwen2", "qwen3", "qwen3_5", "gemma", "gemma2", "phi", "deepseekv2"]


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


def compute_ref_logps_online(model, batch_tensor: dict, batch_info: dict, args, num_seq):
    """Compute reference log-probs online by temporarily disabling LoRA adapters.

    The frozen base model (without LoRA) serves as the reference model.
    Returns per-example sum of log-probs over response tokens (shape: [num_seqs]).
    """
    model.disable_adapter_layers()
    try:
        with torch.no_grad():
            return model(
                **batch_tensor,
                **batch_info,
                num_seq=num_seq,
                return_per_seq_logps=True,
                chunk_size=args.chunk_size,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).logits
    finally:
        model.enable_adapter_layers()


def create_dataset(args, split_name: str) -> NumpyDataset:
    filename = f"{args.data_prefix}.{split_name}"
    if not (os.path.isfile(filename + ".parquet") or os.path.isfile(filename + ".pickle")
            or os.path.isfile(filename + ".part000.parquet") or os.path.isfile(filename + ".part000.pickle")):
        print(f"Skipping loading {split_name}")
        return None
    print(f"Loading {split_name} data from {filename}...")
    return NumpyDataset(filename)


def batch_to_tensor(batch: Dict[str, np.ndarray], dataset_path: Optional[str] = None,
                    processor: Optional[ProcessorMixin] = None, prefix: str = "chosen_"):
    """Collate a batch of sequences (chosen or rejected) into tensors.

    Args:
        batch: Dataset batch with keys prefixed by `prefix` (e.g. 'chosen_seqlens').
        dataset_path: Path to dataset directory (for multimodal image/video resolution).
        processor: Optional processor for multimodal data.
        prefix: 'chosen_' or 'rejected_'.
    """
    images_list = sum([b.tolist() for b in batch.get(f'{prefix}images', [])], start=[])
    videos_list = sum([b.tolist() for b in batch.get(f'{prefix}videos', [])], start=[])

    # Concat sequences
    batch_data = {}
    for k, dtype in _BATCH_KEYS.items():
        key = f"{prefix}{k}"
        if key in batch:
            batch_data[k] = np.concatenate(batch[key], axis=0)

    # To tensor
    batch_tensor: Dict[str, torch.Tensor] = {}
    for k, dtype in _BATCH_KEYS.items():
        if k in batch_data:
            batch_tensor[k] = torch.from_numpy(batch_data[k]).to(dtype)

    if processor is not None and dataset_path is not None:
        if images_list and hasattr(processor, "image_processor") and processor.image_processor is not None:
            images_list = [os.path.join(dataset_path, img) for img in images_list]
            output_images = processor.image_processor(images_list)
            batch_tensor["pixel_values"] = output_images["pixel_values"]
            batch_tensor["image_grid_thw"] = output_images["image_grid_thw"]
        if videos_list and hasattr(processor, "video_processor") and processor.video_processor is not None:
            videos_list = [os.path.join(dataset_path, vid) for vid in videos_list]
            output_videos = processor.video_processor(videos_list)
            batch_tensor["pixel_values_videos"] = output_videos["pixel_values_videos"]
            batch_tensor["video_grid_thw"] = output_videos["video_grid_thw"]

    # cu seqlens
    if "seqlens" in batch_tensor:
        batch_tensor["cu_seqlens"] = torch.nn.functional.pad(
            batch_tensor["seqlens"].cumsum(-1, dtype=torch.int32), (1, 0)
        )
        batch_tensor["max_seqlen"] = torch.max(batch_tensor["seqlens"]).item()
        del batch_tensor["seqlens"]

    # Move max_seqlen out of tensor dict (it's an int, placed in batch_info by collate funcs)
    batch_info = {}
    if "max_seqlen" in batch_tensor:
        batch_info["max_seqlen"] = batch_tensor.pop("max_seqlen")

    return batch_tensor, batch_info


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
    chosen_ref_logps = torch.from_numpy(np.concatenate(batch["chosen_ref_logp"], axis=0)).to(torch.float32)
    rejected_ref_logps = torch.from_numpy(np.concatenate(batch["rejected_ref_logp"], axis=0)).to(torch.float32)

    # Combined batch info
    batch_info = {
        "max_seqlen": max(
            chosen_info.get("max_seqlen", 0),
            rejected_info.get("max_seqlen", 0),
        ),
    }

    return chosen_tensor, rejected_tensor, chosen_ref_logps, rejected_ref_logps, batch_info


def _combine_chosen_rejected_batch(chosen_t: dict, rejected_t: dict) -> tuple:
    """Combine chosen + rejected tensor dicts into a single batch.

    Concatenates packed tensors and offsets the rejected cu_seqlens so that
    sequences from both sides are treated as one batch by the model.

    Returns:
        combined: dict with concatenated tensors.
        num_chosen: number of chosen sequences (for splitting per_seq_logps).
    """
    num_chosen = chosen_t["cu_seqlens"].shape[0] - 1
    chosen_total = chosen_t["cu_seqlens"][-1]

    combined = {}
    for key in chosen_t:
        if key == "cu_seqlens":
            combined[key] = torch.cat([chosen_t[key], rejected_t[key][1:] + chosen_total])
        elif key == "max_seqlen":
            pass  # handled by caller via batch_info
        elif isinstance(chosen_t[key], torch.Tensor):
            combined[key] = torch.cat([chosen_t[key], rejected_t[key]])

    for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
        if key in chosen_t or key in rejected_t:
            c_val = chosen_t.get(key, torch.empty(0, device=next(iter(chosen_t.values())).device))
            r_val = rejected_t.get(key, torch.empty(0, device=next(iter(rejected_t.values())).device))
            if c_val.numel() > 0 and r_val.numel() > 0:
                combined[key] = torch.cat([c_val, r_val])
            elif c_val.numel() > 0:
                combined[key] = c_val
            elif r_val.numel() > 0:
                combined[key] = r_val

    return combined, num_chosen


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


def get_latest_checkpoint(args):
    checkpoint_list = sorted(
        [i for i in Path(args.save_path).glob("checkpoint_*") if i.is_dir()],
        key=lambda x: int(x.name.split("_")[-1]),
    )
    if checkpoint_list:
        return str(checkpoint_list[-1])


def clean_checkpoint(args):
    checkpoint_list = sorted(
        [i for i in Path(args.save_path).glob("checkpoint_*") if i.is_dir()],
        key=lambda x: int(x.name.split("_")[-1]),
    )
    for checkpoint in checkpoint_list[: -args.max_checkpoint]:
        shutil.rmtree(checkpoint, ignore_errors=True)


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def create_lr_scheduler(args, train_total_steps: int):
    return partial(
        cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=args.lr_warmup_step or round(args.lr_warmup_ratio * train_total_steps),
        num_training_steps=train_total_steps,
        min_ratio=args.lr_min_ratio,
    )


def save_tokenizer(args, save_path):
    MODEL_CONFIG_MAP[args.model_type].model_tokenizer_create(args.model_path).save_pretrained(save_path)


def load_tokenizer(args):
    return MODEL_CONFIG_MAP[args.model_type].model_tokenizer_create(args.model_path)


def save_openchat_metadata(args, epoch, latest_step: int, save_path):
    metadata = vars(args)
    metadata["epoch"] = epoch
    metadata["latest_step"] = latest_step
    with open(os.path.join(save_path, "openchat.json"), "w") as f:
        json.dump(metadata, f, default=lambda o: "<non-serializable>")


def calculate_auto_lr(base_lr: float, lr: Optional[float], batch_max_len: int, model_type: str, train_dataset: NumpyDataset):
    if lr is not None:
        return lr

    base_bs = 4_000_000
    if any([x in model_type.lower() for x in MODEL_LR]):
        base_lr /= 6.0

    loss_weights = np.concatenate([
        *train_dataset["chosen_nz_shifted_loss_weights"],
        *train_dataset["rejected_nz_shifted_loss_weights"],
    ])
    supervised_ratio = np.sum(loss_weights != 0) / len(loss_weights)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    supervised_tokens = batch_max_len * world_size * supervised_ratio
    lr = base_lr * math.sqrt(supervised_tokens / base_bs)

    print(f"Use automatic learning rate {lr} (estimated from supervised ratio {supervised_ratio} effective batch size {supervised_tokens})")
    return lr


def mlflow_stopper_wrapper(is_distributed: bool = True):
    def _wrapper(function):
        def mlflow_stopper(args):
            try:
                function(args)
            finally:
                if not is_distributed:
                    mlflow.end_run()
                else:
                    RANK = dist.get_rank()
                    if RANK == 0:
                        mlflow.end_run()
        return mlflow_stopper
    return _wrapper
