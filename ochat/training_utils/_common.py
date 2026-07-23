"""Shared utilities for all training modes (SFT, DPO, ORPO).

Consolidated from duplicated code that previously lived in:
  - ochat/training_sft/utils.py
  - ochat/training_dpo/utils.py
"""

import os
import json
import math
import shutil
import mlflow
import torch
import numpy as np
import torch.distributed as dist

from typing import Dict, Optional
from pathlib import Path
from functools import partial

from transformers import ProcessorMixin

from ochat.config import MODEL_CONFIG_MAP
from ochat.training_utils.numpy_dataset import NumpyDataset


BATCH_KEYS = {
    "seqlens": torch.long,
    "nz_input_ids": torch.long,
    "nz_position_ids": torch.long,
    "nz_shifted_label_ids": torch.long,
    "nz_shifted_loss_weights": torch.bfloat16,
}

MODEL_LR = ["mistral", "mixtral", "qwen2", "qwen3", "qwen3_5", "gemma", "gemma2", "phi", "deepseekv2"]


# -- Dataset loading ----------------------------------------------------------

def create_dataset(args, split_name: str) -> Optional[NumpyDataset]:
    filename = f"{args.data_prefix}.{split_name}"
    if not (os.path.isfile(filename + ".parquet") or os.path.isfile(filename + ".pickle")
            or os.path.isfile(filename + ".part000.parquet") or os.path.isfile(filename + ".part000.pickle")):
        print(f"Skipping loading {split_name}")
        return None
    print(f"Loading {split_name} data from {filename}...")
    return NumpyDataset(filename)


# -- Batch collation (shared by SFT, DPO, ORPO) -------------------------------


def combine_chosen_rejected_batch(chosen_t: dict, rejected_t: dict) -> tuple:
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


def batch_to_tensor(batch: Dict[str, np.ndarray], dataset_path: Optional[str] = None,
                    processor: Optional[ProcessorMixin] = None, prefix: str = ""):
    """Collate a batch of sequences into tensors.

    Args:
        batch: Dataset batch with keys optionally prefixed (e.g. 'chosen_seqlens' for DPO/ORPO).
        dataset_path: Path to dataset directory (for multimodal image/video resolution).
        processor: Optional processor for multimodal data.
        prefix: Key prefix for DPO/ORPO paired data ('chosen_', 'rejected_'). Empty for SFT.
    """
    images_list = sum([b.tolist() for b in batch.get(f'{prefix}images', [])], start=[])
    videos_list = sum([b.tolist() for b in batch.get(f'{prefix}videos', [])], start=[])

    # Concat sequences
    batch_data = {}
    for k in BATCH_KEYS:
        key = f"{prefix}{k}"
        if key in batch:
            batch_data[k] = np.concatenate(batch[key], axis=0)

    # To tensor
    batch_tensor: Dict[str, torch.Tensor] = {}
    for k, dtype in BATCH_KEYS.items():
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


# -- Checkpointing ------------------------------------------------------------

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


# -- LR scheduling ------------------------------------------------------------

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


# -- Tokenizer helpers --------------------------------------------------------

def save_tokenizer(args, save_path):
    MODEL_CONFIG_MAP[args.model_type].model_tokenizer_create(args.model_path).save_pretrained(save_path)


def load_tokenizer(args):
    return MODEL_CONFIG_MAP[args.model_type].model_tokenizer_create(args.model_path)


# -- Metadata -----------------------------------------------------------------

def save_openchat_metadata(args, epoch, latest_step: int, save_path):
    metadata = vars(args)
    metadata["epoch"] = epoch
    metadata["latest_step"] = latest_step
    with open(os.path.join(save_path, "openchat.json"), "w") as f:
        json.dump(metadata, f, default=lambda o: "<non-serializable>")


# -- Auto LR estimation -------------------------------------------------------

def calculate_auto_lr(base_lr: float, lr: Optional[float], batch_max_len: int,
                      model_type: str, train_dataset: NumpyDataset):
    if lr is not None:
        return lr

    base_bs = 4_000_000
    if any(x in model_type.lower() for x in MODEL_LR):
        base_lr /= 6.0

    # Auto-detect SFT vs DPO/ORPO dataset format
    if "chosen_nz_shifted_loss_weights" in train_dataset.dataset:
        loss_weights = np.concatenate([
            *train_dataset["chosen_nz_shifted_loss_weights"],
            *train_dataset["rejected_nz_shifted_loss_weights"],
        ])
    else:
        loss_weights = np.concatenate(train_dataset["nz_shifted_loss_weights"])

    supervised_ratio = np.sum(loss_weights != 0) / len(loss_weights)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    supervised_tokens = batch_max_len * world_size * supervised_ratio
    lr = base_lr * math.sqrt(supervised_tokens / base_bs)

    print(f"Use automatic learning rate {lr} (estimated from supervised ratio {supervised_ratio} effective batch size {supervised_tokens})")
    return lr


# -- MLflow -------------------------------------------------------------------

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
