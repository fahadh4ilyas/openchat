import argparse
import os
import math
import json
import shutil
from pathlib import Path
from functools import partial
from typing import Optional, Union, Literal, Tuple, Dict

from pydantic import BaseModel, Field, validator

import torch
import torch.distributed as dist

import tqdm
import mlflow
import numpy as np

from sentence_transformers import SentenceTransformer

from transformers.integrations import HfDeepSpeedConfig

try:
    import deepspeed
except ImportError:
    raise ImportError("Please install deepspeed to train models.")

class TrainingArguments(BaseModel):

    local_rank: int = Field(...)
    model_path: str = Field(...)
    model_type: Optional[str] = Field(None)
    data_prefix: str = Field(...)
    save_path: str = Field(...)
    save_every: Optional[int] = Field(None, gt=0)
    save_strategy: Union[Literal['epoch'], Literal['step']] = Field('epoch')
    checkpoint_every: int = Field(0, ge=0)
    max_checkpoint: int = Field(1, gt=0)
    eval_every: Optional[int] = Field(None, gt=0)
    eval_strategy: Union[Literal['epoch'], Literal['step']] = Field('epoch')
    batch_max_len: int = Field(81920)
    epochs: int = Field(5)
    max_steps: int = Field(0)
    use_zero_one_opt: bool = Field(False)
    base_lr: float = Field(3e-4)
    lr: Optional[float] = Field(None)
    lr_min_ratio: float = Field(0.1)
    lr_warmup_ratio: float = Field(0.05)
    lr_warmup_step: int = Field(0)
    weight_decay: float = Field(0.1)
    beta1: float = Field(0.9)
    beta2: float = Field(0.95)
    eps: float = Field(1e-5)
    tracking_uri: Optional[str] = Field(None)
    mlflow_username: Optional[str] = Field(None)
    mlflow_password: Optional[str] = Field(None)
    experiment_name: str = Field(...)
    run_name: str = Field(...)
    deepscale: bool = Field(False)
    deepscale_config: Optional[str] = Field(None)
    deepspeed: bool = Field(True)
    deepspeed_config: Union[str, dict] = Field(...)
    deepspeed_mpi: bool = Field(False)
    ds_offload: bool = Field(False)
    ds_zero_op: int = Field(0)
    device: Optional[str] = Field(None)

def get_latest_checkpoint(args: TrainingArguments):

    checkpoint_list = sorted([i for i in Path(args.save_path).glob('checkpoint_*') if i.is_dir()], key=lambda x: int(x.name.split('_')[-1]))
    if checkpoint_list:
        return str(checkpoint_list[-1])

def clean_checkpoint(args: TrainingArguments):

    checkpoint_list = sorted([i for i in Path(args.save_path).glob('checkpoint_*') if i.is_dir()], key=lambda x: int(x.name.split('_')[-1]))
    
    for checkpoint in checkpoint_list[:-args.max_checkpoint]:
        shutil.rmtree(checkpoint, ignore_errors=True)

def create_model(args: TrainingArguments):
    print(f"Loading model {args.model_type} from {args.model_path}...")

    # get checkpoint
    model_path = get_latest_checkpoint(args) or args.model_path

    # Create model + optimizer + lr scheduler
    model = SentenceTransformer(model_path, model_kwargs=dict(low_cpu_mem_usage=args.ds_zero_op != 3, trust_remote_code = True, ))
    if not args.ds_offload:
        # Model to assigned cuda device
        model = model.to(args.local_rank)
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Optimizer
    if args.ds_offload:
        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(model.parameters(),
                                                        lr=args.lr,
                                                        weight_decay=args.weight_decay,
                                                        betas=(args.beta1, args.beta2),
                                                        eps=args.eps)
    elif args.use_zero_one_opt:
        with open(args.deepspeed_config) as f:
            ds_config: dict = json.load(f)
        ds_config['optimizer'] = {
            "type": "ZeroOneAdam",
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "betas": [args.beta1, args.beta2],
                "eps": args.eps
            }
        }
        ds_config.pop("zero_optimization", None)
        args.deepspeed_config = ds_config
        optimizer = None
    else:
        optimizer = deepspeed.ops.adam.FusedAdam(model.parameters(),
                                             lr=args.lr,
                                             weight_decay=args.weight_decay,
                                             betas=(args.beta1, args.beta2),
                                             eps=args.eps)

    # DeepSpeed model
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters(),
                                                         optimizer=optimizer)

    # Put deepspeed arguments
    args.device                         = model_engine.device

    return model_engine, optimizer


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def create_lr_scheduler(args: TrainingArguments, train_total_steps: int):
    lr_scheduler = partial(
        cosine_schedule_with_warmup_lr_lambda,

        num_warmup_steps=args.lr_warmup_step or round(args.lr_warmup_ratio * train_total_steps),
        num_training_steps=train_total_steps,
        min_ratio=args.lr_min_ratio
    )

    return lr_scheduler


def save_tokenizer(args: TrainingArguments, save_path):
    MODEL_CONFIG_MAP[args.model_type].model_tokenizer_create(args.model_path).save_pretrained(save_path)


def save_openchat_metadata(args: TrainingArguments, epoch: Union[int, float], latest_step: int, save_path):
    metadata = vars(args)
    metadata["epoch"] = epoch
    metadata["latest_step"] = latest_step

    with open(os.path.join(save_path, "openchat.json"), "w") as f:
        json.dump(metadata, f, default=lambda o: "<non-serializable>")


def train(args: TrainingArguments):
    deepspeed.init_distributed(dist_backend="nccl")
    dsconfig = HfDeepSpeedConfig(args.deepspeed_config)
    RANK = dist.get_rank()

