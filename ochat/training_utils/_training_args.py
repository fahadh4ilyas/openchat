"""Shared training argument definitions for all training entry points.

Avoids duplication across train.py, train_ring.py, train_lora.py, train_ring_lora.py.
"""

import argparse
from typing import Optional, Union, Literal, List

from pydantic import BaseModel, Field, field_validator as validator


class BaseTrainingArguments(BaseModel):
    """Arguments shared by all training modes (full, LoRA, ring, etc.)."""

    local_rank: Optional[int] = Field(None)
    model_path: str = Field(...)
    model_type: Optional[str] = Field(None)
    has_processor: Optional[bool] = Field(None)
    data_prefix: str = Field(...)
    save_path: str = Field(...)
    save_every: Optional[int] = Field(None, gt=0)
    save_strategy: Union[Literal["epoch"], Literal["step"]] = Field("epoch")
    checkpoint_every: int = Field(0, ge=0)
    max_checkpoint: int = Field(1, gt=0)
    eval_every: Optional[int] = Field(None, gt=0)
    eval_strategy: Union[Literal["epoch"], Literal["step"]] = Field("epoch")
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
    use_fast_norm: bool = Field(False)
    use_fast_rope: bool = Field(False)
    torch_empty_cache_steps: Optional[int] = Field(None, gt=0)
    tracking_uri: Optional[str] = Field(None)
    mlflow_username: Optional[str] = Field(None)
    mlflow_password: Optional[str] = Field(None)
    experiment_name: str = Field(...)
    run_name: str = Field(...)
    deepscale: bool = Field(False)
    deepscale_config: Optional[str] = Field(None)
    deepspeed: bool = Field(False)
    deepspeed_config: Optional[Union[str, dict]] = Field(None)
    deepspeed_mpi: bool = Field(False)
    ds_offload: bool = Field(False)
    ds_zero_op: int = Field(0)
    device: Optional[str] = Field(None)

    @validator("batch_max_len")
    def val_batch_size(cls, v: int) -> int:
        if v % 2048 != 0:
            raise ValueError("`batch_max_len` must be multiple of 2048")
        return v


class LoraTrainingArgsMixin(BaseModel):
    """LoRA-specific arguments. Mix into TrainingArguments classes."""

    use_lora: bool = Field(False)
    lora_alpha: int = Field(32)
    lora_r: int = Field(32)
    lora_dropout: float = Field(0.05)
    lora_target_modules: List[str] = Field(["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_exclude_modules: Optional[List[str]] = Field(None)
    lora_bias: str = Field("none")
    lora_fan_in_fan_out: bool = Field(False)
    lora_use_rslora: bool = Field(False)
    lora_use_dora: bool = Field(False)
    lora_use_qalora: bool = Field(False)
    lora_qalora_group_size: int = Field(16)
    lora_modules_to_save: Optional[List[str]] = Field(None)
    use_qlora: bool = Field(False)
    quant_bits: int = Field(4)
    quant_type_4bit: str = Field("nf4")
    use_double_quant_4bit: bool = Field(False)


# -- Argparse helpers --------------------------------------------------------

def add_base_args(parser: argparse.ArgumentParser, base_lr: float = 3e-4):
    """Add common training arguments to an argparse parser."""
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--data-prefix", "--data_prefix", type=str, required=True)
    parser.add_argument("--save-path", "--save_path", type=str, required=True)
    parser.add_argument("--save-strategy", "--save_strategy", type=str, choices=["epoch", "step"], default="epoch")
    parser.add_argument("--save-every", "--save_every", type=int, default=None)
    parser.add_argument("--checkpoint-every", "--checkpoint_every", type=int, default=0)
    parser.add_argument("--max-checkpoint", "--max_checkpoint", type=int, default=1)
    parser.add_argument("--eval-strategy", "--eval_strategy", type=str, choices=["epoch", "step"], default="epoch")
    parser.add_argument("--eval-every", "--eval_every", type=int, default=None)
    parser.add_argument("--batch-max-len", "--batch_max_len", type=int, default=81920)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", "--max_steps", type=int, default=0)
    parser.add_argument("--use-zero-one-opt", "--use_zero_one_opt", action="store_true")
    parser.add_argument("--base-lr", "--base_lr", type=float, default=base_lr)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-min-ratio", "--lr_min_ratio", type=float, default=0.1)
    parser.add_argument("--lr-warmup-ratio", "--lr_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr-warmup-step", "--lr_warmup_step", type=int, default=0)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--use-fast-norm", "--use_fast_norm", action="store_true")
    parser.add_argument("--use-fast-rope", "--use_fast_rope", action="store_true")
    parser.add_argument("--torch-empty-cache-steps", "--torch_empty_cache_steps", type=int, default=None)
    parser.add_argument("--tracking-uri", "--tracking_uri", type=str, default=None)
    parser.add_argument("--mlflow-username", "--mlflow_username", type=str, default=None)
    parser.add_argument("--mlflow-password", "--mlflow_password", type=str, default=None)
    parser.add_argument("--experiment-name", "--experiment_name", type=str, required=True)
    parser.add_argument("--run-name", "--run_name", type=str, required=True)


def add_lora_args(parser: argparse.ArgumentParser):
    """Add LoRA-specific arguments to an argparse parser."""
    parser.add_argument("--use-lora", "--use_lora", action="store_true")
    parser.add_argument("--lora-alpha", "--lora_alpha", type=int, default=32)
    parser.add_argument("--lora-r", "--lora_r", type=int, default=32)
    parser.add_argument("--lora-dropout", "--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", "--lora_target_modules", type=str, nargs="*", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--lora-exclude-modules", "--lora_exclude_modules", type=str, nargs="*", default=None)
    parser.add_argument("--lora-bias", "--lora_bias", type=str, default="none")
    parser.add_argument("--lora-fan-in-fan-out", "--lora_fan_in_fan_out", action="store_true")
    parser.add_argument("--lora-use-rslora", "--lora_use_rslora", action="store_true")
    parser.add_argument("--lora-use-dora", "--lora_use_dora", action="store_true")
    parser.add_argument("--lora-use-qalora", "--lora_use_qalora", action="store_true")
    parser.add_argument("--lora-qalora-group-size", "--lora_qalora_group_size", type=int, default=16)
    parser.add_argument("--lora-modules-to-save", "--lora_modules_to_save", type=str, nargs="*", default=None)
    parser.add_argument("--use-qlora", "--use_qlora", action="store_true")
    parser.add_argument("--quant-bits", "--quant_bits", type=int, default=4)
    parser.add_argument("--quant-type-4bit", "--quant_type_4bit", type=str, default="nf4")
    parser.add_argument("--use-double-quant-4bit", "--use_double_quant_4bit", action="store_true")
