import os
import json
import math
import shutil
import mlflow
import torch
import numpy as np
import torch.distributed as dist

from typing import Dict, Union, Optional
from pathlib import Path
from functools import partial

from ochat.config import MODEL_CONFIG_MAP
from ochat.training_deepspeed.numpy_dataset import NumpyDataset


PAD_ID = 0
BATCH_KEYS = {
    "seqlens": torch.long,
    "nz_input_ids": torch.long,
    "nz_position_ids": torch.long,
    "nz_shifted_label_ids": torch.long,
    "nz_shifted_loss_weights": torch.bfloat16,
}


MODEL_LR = ["mistral", "qwen2", "mixtral"]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def create_dataset(args, split_name: str) -> NumpyDataset:
    # Load data
    filename = f"{args.data_prefix}.{split_name}"
    if not os.path.isfile(filename + ".parquet") and not os.path.isfile(filename + ".pickle"):
        print(f"Skipping loading {split_name}")
        return None

    print(f"Loading {split_name} data from {filename}...")
    return NumpyDataset(filename)


def batch_to_tensor(batch: Dict[str, np.ndarray]):
    # Concat batches
    batch = {k: np.concatenate(batch[k], axis=0) for k in BATCH_KEYS.keys()}

    # Pad an unused item to reach multiple of 64, for faster GEMM
    total_seqlen = batch["nz_input_ids"].size
    pad_len = _find_multiple(total_seqlen, 64) - total_seqlen

    if pad_len > 0:
        assert pad_len < 64

        # total length
        padding_specs = {
            "seqlens": (1, pad_len),
            "nz_input_ids": (pad_len, PAD_ID),
            "nz_position_ids": (pad_len, 0),
            "nz_shifted_label_ids": (pad_len, PAD_ID),
            "nz_shifted_loss_weights": (pad_len, 0),
        }
        for k, pad_spec in padding_specs.items():
            batch[k] = np.concatenate(
                (batch[k], np.full(*pad_spec, dtype=batch[k].dtype)), axis=0
            )

    # to tensor
    batch_tensor: Dict[str, torch.Tensor] = {}
    for k, dtype in BATCH_KEYS.items():
        batch_tensor[k] = torch.from_numpy(batch[k]).to(dtype)

    # cu seqlens
    batch_tensor["cu_seqlens"] = torch.nn.functional.pad(
        batch_tensor["seqlens"].cumsum(-1, dtype=torch.int32), (1, 0)
    )
    # batch info
    batch_info = {"max_seqlen": torch.max(batch_tensor["seqlens"]).item()}

    # inputs
    del batch_tensor["seqlens"]
    return batch_tensor, batch_info


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

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return min_ratio + max(
        0.0,
        (1 - min_ratio)
        * 0.5
        * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
    )


def create_lr_scheduler(args, train_total_steps: int):
    lr_scheduler = partial(
        cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=args.lr_warmup_step
        or round(args.lr_warmup_ratio * train_total_steps),
        num_training_steps=train_total_steps,
        min_ratio=args.lr_min_ratio,
    )

    return lr_scheduler


def save_tokenizer(args, save_path):
    MODEL_CONFIG_MAP[args.model_type].model_tokenizer_create(
        args.model_path
    ).save_pretrained(save_path)


def save_openchat_metadata(
    args, epoch: Union[int, float], latest_step: int, save_path
):
    metadata = vars(args)
    metadata["epoch"] = epoch
    metadata["latest_step"] = latest_step

    with open(os.path.join(save_path, "openchat.json"), "w") as f:
        json.dump(metadata, f, default=lambda o: "<non-serializable>")


def calculate_auto_lr(
    base_lr: float,
    lr: Optional[float],
    batch_max_len: int,
    model_type: str,
    train_dataset: NumpyDataset,
):
    if lr is not None:
        return lr

    # Llama hyperparameters
    # FIXME: Only 7B/13B is supported
    base_bs = 4_000_000
    if any([x in model_type.lower() for x in MODEL_LR]):
        base_lr /= 6.0

    loss_weights = np.concatenate(train_dataset["nz_shifted_loss_weights"])
    supervised_ratio = np.sum(loss_weights != 0) / len(loss_weights)

    supervised_tokens = batch_max_len * dist.get_world_size() * supervised_ratio
    lr = base_lr * math.sqrt(supervised_tokens / base_bs)

    print(
        f"Use automatic learning rate {lr} (estimated from supervised ratio {supervised_ratio} effective batch size {supervised_tokens})"
    )
    return lr


def mlflow_stopper_wrapper(function):

    def mlflow_stopper(args):

        try:
            function(args)
        except:
            raise
        finally:
            RANK = dist.get_rank()
            if RANK == 0:
                mlflow.end_run()
    
    return mlflow_stopper