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
    lora_bias: str = Field("none")
    modules_to_save: Optional[List[str]] = Field(None)
    use_qlora: bool = Field(False)
    quant_bits: int = Field(4)
    quant_type_4bit: str = Field("nf4")
    use_double_quant_4bit: bool = Field(False)


# -- Argparse helpers --------------------------------------------------------

def add_base_args(parser: argparse.ArgumentParser, base_lr: float = 3e-4):
    """Add common training arguments to an argparse parser."""
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-prefix", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--save-strategy", type=str, choices=["epoch", "step"], default="epoch")
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--max-checkpoint", type=int, default=1)
    parser.add_argument("--eval-strategy", type=str, choices=["epoch", "step"], default="epoch")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--batch-max-len", type=int, default=81920)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--use-zero-one-opt", action="store_true")
    parser.add_argument("--base-lr", type=float, default=base_lr)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-warmup-step", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--use-fast-norm", action="store_true")
    parser.add_argument("--use-fast-rope", action="store_true")
    parser.add_argument("--torch-empty-cache-steps", type=int, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-username", type=str, default=None)
    parser.add_argument("--mlflow-password", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)


def add_lora_args(parser: argparse.ArgumentParser):
    """Add LoRA-specific arguments to an argparse parser."""
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, nargs="*", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--lora-bias", type=str, default="none")
    parser.add_argument("--modules-to-save", type=str, nargs="*", default=None)
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--quant-bits", type=int, default=4)
    parser.add_argument("--quant-type-4bit", type=str, default="nf4")
    parser.add_argument("--use-double-quant-4bit", action="store_true")
