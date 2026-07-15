import argparse
import os
import json
from functools import partial
from typing import Optional, Union, Literal, List

from pydantic import BaseModel, Field, field_validator as validator

import torch
import torch.distributed as dist

import tqdm
import mlflow

from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

from ochat.config import MODEL_CONFIG_MAP
from ochat.training_sft.utils import (
    mlflow_stopper_wrapper,
    batch_to_tensor,
    get_latest_checkpoint,
    create_dataset,
    calculate_auto_lr,
    create_lr_scheduler,
    save_openchat_metadata,
    clean_checkpoint,
    save_tokenizer,
    load_tokenizer,
)
from ochat.training_sft.multipack_dataloader_ring import MultipackDistributedDataloader
from ochat.training_sft.numpy_dataset import NumpyDataset

from transformers.integrations import HfDeepSpeedConfig
from transformers import BitsAndBytesConfig

try:
    import deepspeed
except ImportError:
    raise ImportError("Please install deepspeed to train models.")


class TrainingArguments(BaseModel):
    local_rank: int = Field(...)
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
    base_lr: float = Field(1e-2)
    lr: Optional[float] = Field(None)
    lr_min_ratio: float = Field(0.1)
    lr_warmup_ratio: float = Field(0.05)
    lr_warmup_step: int = Field(0)
    weight_decay: float = Field(0.1)
    beta1: float = Field(0.9)
    beta2: float = Field(0.95)
    eps: float = Field(1e-5)
    chunk_size: int = Field(-1)
    use_fast_norm: bool = Field(False)
    use_fast_rope: bool = Field(False)
    torch_empty_cache_steps: Optional[int] = Field(None, gt=0)
    tracking_uri: Optional[str] = Field(None)
    mlflow_username: Optional[str] = Field(None)
    mlflow_password: Optional[str] = Field(None)
    experiment_name: str = Field(...)
    run_name: str = Field(...)
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
    deepscale: bool = Field(False)
    deepscale_config: Optional[str] = Field(None)
    deepspeed: bool = Field(True)
    deepspeed_config: Union[str, dict] = Field(...)
    deepspeed_mpi: bool = Field(False)
    ds_offload: bool = Field(False)
    ds_zero_op: int = Field(0)
    device: Optional[str] = Field(None)

    @validator("batch_max_len")
    def val_batch_size(cls, v: int) -> int:
        if v % 2048 != 0:
            raise ValueError("`batch_max_len` must be multiple of 2048")

        return v


def parse_args() -> argparse.Namespace:
    parser_base = argparse.ArgumentParser()
    # Distributed
    parser_base.add_argument("--local_rank", type=int, required=True)

    # Model type and data
    parser_base.add_argument("--model_path", type=str, required=True)
    parser_base.add_argument("--data_prefix", type=str, required=True)
    parser_base.add_argument("--save_path", type=str, required=True)
    parser_base.add_argument(
        "--save_strategy", type=str, choices=["epoch", "step"], default="epoch"
    )
    parser_base.add_argument("--save_every", type=int, default=None)
    parser_base.add_argument("--checkpoint_every", type=int, default=0)
    parser_base.add_argument("--max_checkpoint", type=int, default=1)
    parser_base.add_argument(
        "--eval_strategy", type=str, choices=["epoch", "step"], default="epoch"
    )
    parser_base.add_argument("--eval_every", type=int, default=None)

    # Hyperparameters
    parser_base.add_argument("--batch_max_len", type=int, default=81920)
    parser_base.add_argument("--epochs", type=int, default=5)
    parser_base.add_argument("--max_steps", type=int, default=0)

    # Set lr to None to automatically estimate from LLaMA pretraining parameters (e.g. lr ~ sqrt(batch_size))
    parser_base.add_argument("--use_zero_one_opt", action="store_true")
    parser_base.add_argument("--base_lr", type=float, default=1e-2)
    parser_base.add_argument("--lr", type=float, default=None)
    parser_base.add_argument("--lr_min_ratio", type=float, default=0.1)
    parser_base.add_argument("--lr_warmup_ratio", type=float, default=0.05)
    parser_base.add_argument("--lr_warmup_step", type=int, default=0)

    parser_base.add_argument("--weight_decay", type=float, default=0.1)

    parser_base.add_argument("--beta1", type=float, default=0.9)
    parser_base.add_argument("--beta2", type=float, default=0.95)
    parser_base.add_argument("--eps", type=float, default=1e-5)

    # CHUNKING
    parser_base.add_argument("--chunk_size", type=int, default=-1)

    # FAST FORWARD
    parser_base.add_argument("--use_fast_norm", action="store_true")
    parser_base.add_argument("--use_fast_rope", action="store_true")

    # CACHING
    parser_base.add_argument(
        "--torch_empty_cache_steps", type=int, default=None
    )

    # MLFLOW
    parser_base.add_argument("--tracking_uri", type=str, default=None)
    parser_base.add_argument("--mlflow_username", type=str, default=None)
    parser_base.add_argument("--mlflow_password", type=str, default=None)
    parser_base.add_argument("--experiment_name", type=str, required=True)
    parser_base.add_argument("--run_name", type=str, required=True)

    # LORA
    parser_base.add_argument("--lora_alpha", type=int, default=32)
    parser_base.add_argument("--lora_r", type=int, default=32)
    parser_base.add_argument("--lora_dropout", type=float, default=0.05)
    parser_base.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser_base.add_argument("--lora_bias", type=str, default="none")
    parser_base.add_argument("--modules_to_save", type=str, nargs="*", default=None)

    # QLORA
    parser_base.add_argument("--use_qlora", action="store_true")
    parser_base.add_argument("--quant_bits", type=int, default=4)
    parser_base.add_argument("--quant_type_4bit", type=str, default="nf4")
    parser_base.add_argument("--use_double_quant_4bit", action="store_true")

    # DeepSpeed parameters
    parser_base = deepspeed.add_config_arguments(parser_base)

    # Parse known args
    args_base = parser_base.parse_args()
    return args_base


def create_distributed_dataloader(args: TrainingArguments, data: NumpyDataset):
    collate_fn = batch_to_tensor
    if args.has_processor:
        tokenizer = load_tokenizer(args)
        collate_fn = partial(batch_to_tensor, dataset_path=os.path.dirname(args.data_prefix), processor=tokenizer)
    # Multipack dataloader
    return MultipackDistributedDataloader(
        dataset=data,
        lengths=data["total_length"],
        batch_max_length=args.batch_max_len,
        collate_fn=collate_fn,
        seed=0,
    )


def create_model(args: TrainingArguments):
    print(f"Loading model {args.model_type} from {args.model_path}...")

    # get checkpoint
    model_path = get_latest_checkpoint(args) or args.model_path

    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.quant_bits == 8,
            load_in_4bit=args.quant_bits == 4,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=args.quant_type_4bit,
            bnb_4bit_use_double_quant=args.use_double_quant_4bit,
        )

    # Create model + optimizer + lr scheduler
    if model_path == args.model_path:
        model = (
            MODEL_CONFIG_MAP[args.model_type]
            .model_create_for_training(
                model_path,
                low_cpu_mem_usage=args.ds_zero_op != 3,
                quantization_config=quantization_config,
            )
            .to(args.local_rank)
        )
        model.config.use_cache = False
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model)
        # Create lora config
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            modules_to_save=args.modules_to_save,
        )
        # Create Lora Model
        model = get_peft_model(model, lora_config)
    else:
        model = (
            MODEL_CONFIG_MAP[args.model_type]
            .model_create_for_training(
                args.model_path,
                low_cpu_mem_usage=args.ds_zero_op != 3,
                quantization_config=quantization_config,
            )
            .to(args.local_rank)
        )
        model.config.use_cache = False
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
    if not args.ds_offload:
        # Model to assigned cuda device
        model = model.to(args.local_rank)
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()

    # Optimizer
    if args.ds_offload:
        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
    elif args.use_zero_one_opt:
        with open(args.deepspeed_config) as f:
            ds_config: dict = json.load(f)
        ds_config["optimizer"] = {
            "type": "ZeroOneAdam",
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "betas": [args.beta1, args.beta2],
                "eps": args.eps,
            },
        }
        ds_config.pop("zero_optimization", None)
        args.deepspeed_config = ds_config
        optimizer = None
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            fused=True
        )

    # DeepSpeed model
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(), optimizer=optimizer
    )

    # Put deepspeed arguments
    args.device = model_engine.device

    return model_engine, optimizer


@mlflow_stopper_wrapper()
def train(args: TrainingArguments):
    deepspeed.init_distributed(dist_backend="nccl")
    dsconfig = HfDeepSpeedConfig(args.deepspeed_config)
    RANK = dist.get_rank()

    # Dataset
    train_dataset = create_dataset(args, "train")
    eval_dataset = create_dataset(args, "eval")

    if train_dataset is None:
        raise RuntimeError("Training data not found.")

    # Load model type
    args.model_type = train_dataset.metadata["model_type"]
    args.has_processor = MODEL_CONFIG_MAP[args.model_type].model_has_processor

    # Data Loader
    train_loader = create_distributed_dataloader(args, train_dataset)
    if args.max_steps > 0:
        args.epochs = -(-args.max_steps // train_loader.num_batches())
        train_total_steps = args.max_steps
    else:
        train_total_steps = args.epochs * train_loader.num_batches()

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = create_distributed_dataloader(args, eval_dataset)

    # Hyperparams
    args.lr = calculate_auto_lr(
        args.base_lr, args.lr, args.batch_max_len, args.model_type, train_dataset
    )

    # Logger
    if RANK == 0:
        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        if args.mlflow_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = args.mlflow_username
        if args.mlflow_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = args.mlflow_password
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run(run_name=args.run_name)
        metadata = vars(args).copy()
        metadata.pop("local_rank", None)
        metadata.pop("device", None)
        metadata["steps"] = train_total_steps
        mlflow.log_params(metadata)

    # Model
    model_engine, optimizer = create_model(args)

    # LR Scheduler
    lr_scheduler = create_lr_scheduler(args, train_total_steps)

    # Progress bar
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_total_steps)

    # Training Loop
    step = 0
    latest_checkpoint = int((get_latest_checkpoint(args) or "_0").split("_")[-1])
    lr_this_step = None
    model_engine.train()
    eval_epoch = 0
    for epoch in range(args.epochs):
        print(f"[rank {RANK}]: Epoch {epoch}")

        train_loader.set_epoch(epoch)
        for (batch_tensor, batch_info), total_seqs in train_loader:
            step += 1
            if step > train_total_steps:  # At most train_total_steps
                break
            elif step <= latest_checkpoint:
                if RANK == 0:
                    progress_bar.update()
                continue

            # To device
            batch_tensor = {
                k: (v.to(args.device) if v is not None else None)
                for k, v in batch_tensor.items()
            }

            # Update
            loss, acc = model_engine(
                **batch_tensor,
                **batch_info,
                total_seqs=total_seqs,
                chunk_size=args.chunk_size,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).loss

            if isinstance(loss, tuple):
                loss, _ = loss

            model_engine.backward(loss)

            if model_engine.is_gradient_accumulation_boundary():
                # Set LR
                lr_this_step = args.lr * lr_scheduler(step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_this_step

            model_engine.step()

            dist.reduce(loss, 0)
            dist.reduce(acc, 0)

            del batch_tensor
            if args.torch_empty_cache_steps is not None and step % args.torch_empty_cache_steps == 0:
                torch.cuda.empty_cache()

            # Logging
            if RANK == 0:
                mlflow.log_metrics(
                    metrics={
                        "train/loss": loss.item(),
                        "train/acc": acc.item(),
                        "train/lr": lr_this_step,
                        "train/epoch": args.epochs * step / train_total_steps,
                    },
                    step=step,
                )
                progress_bar.update()  # type: ignore

            if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0):
                dist.barrier()

                if model_engine.zero_optimization_stage() == 3:
                    state_dict = model_engine._zero3_consolidated_16bit_state_dict()
                elif RANK == 0:
                    state_dict = (
                        deepspeed.checkpoint.utils.clone_tensors_for_torch_save(
                            model_engine.module.state_dict()
                        )
                    )

                if RANK == 0:
                    save_path = os.path.join(args.save_path, f"checkpoint_{step}")

                    model_engine.module.save_pretrained(
                        save_path, state_dict=state_dict
                    )  # type: ignore

                    # Write metadata
                    save_openchat_metadata(args, epoch + 1, step, save_path)

                    clean_checkpoint(args)

            if eval_loader is not None and (
                args.eval_strategy == "step"
                and args.eval_every
                and (step % args.eval_every == 0)
            ):
                model_engine.eval()

                eval_total_metric = torch.zeros(
                    (2,), dtype=torch.float32, device=args.device
                )
                eval_total_steps = 0

                eval_loader.set_epoch(eval_epoch)
                with torch.inference_mode():
                    for (batch_tensor, batch_info), total_seqs in eval_loader:
                        # To device
                        batch_tensor = {
                            k: (v.to(args.device) if v is not None else None)
                            for k, v in batch_tensor.items()
                        }

                        # Eval
                        eval_loss, eval_acc = model_engine(
                            **batch_tensor,
                            **batch_info,
                            total_seqs=total_seqs,
                            chunk_size=args.chunk_size,
                            use_fast_norm=args.use_fast_norm,
                            use_fast_rope=args.use_fast_rope,
                        ).loss

                        if isinstance(eval_loss, tuple):
                            eval_loss, _ = eval_loss

                        # Accumulate eval loss
                        eval_total_metric.add_(torch.stack([eval_loss, eval_acc]))
                        eval_total_steps += 1

                # Gather eval loss (reduce sum)
                eval_total_metric.div_(eval_total_steps)
                dist.reduce(eval_total_metric, 0)

                eval_epoch += 1

                if RANK == 0:
                    eval_loss, eval_acc = eval_total_metric.cpu().numpy()
                    mlflow.log_metrics(
                        metrics={"eval/loss": eval_loss, "eval/acc": eval_acc},
                        step=step,
                    )

                model_engine.train()

            if (
                args.save_strategy == "step"
                and args.save_every
                and (step % args.save_every == 0)
            ):
                dist.barrier()

                if model_engine.zero_optimization_stage() == 3:
                    state_dict = model_engine._zero3_consolidated_16bit_state_dict()
                elif RANK == 0:
                    state_dict = (
                        deepspeed.checkpoint.utils.clone_tensors_for_torch_save(
                            model_engine.module.state_dict()
                        )
                    )

                if RANK == 0:
                    save_path = os.path.join(args.save_path, f"st_{step}")

                    model_engine.module.save_pretrained(
                        save_path, state_dict=state_dict
                    )  # type: ignore

                    # Also save tokenizer from base model
                    save_tokenizer(args, save_path)

                    # Write metadata
                    save_openchat_metadata(
                        args, args.epochs * step / train_total_steps, step, save_path
                    )

        if step > latest_checkpoint:
            # Log batch efficiency
            if RANK == 0:
                mlflow.log_metrics(
                    metrics={"batch_efficiency": train_loader.efficiency()}, step=step
                )

            if eval_loader is not None and (
                (step == train_total_steps)
                or (epoch + 1 == args.epochs)
                or (
                    args.eval_strategy == "epoch"
                    and args.eval_every
                    and ((epoch + 1) % args.eval_every == 0)
                )
            ):
                model_engine.eval()

                eval_total_metric = torch.zeros(
                    (2,), dtype=torch.float32, device=args.device
                )
                eval_total_steps = 0

                eval_loader.set_epoch(eval_epoch)
                with torch.inference_mode():
                    for (batch_tensor, batch_info), total_seqs in eval_loader:
                        # To device
                        batch_tensor = {
                            k: (v.to(args.device) if v is not None else None)
                            for k, v in batch_tensor.items()
                        }

                        # Eval
                        eval_loss, eval_acc = model_engine(
                            **batch_tensor,
                            **batch_info,
                            total_seqs=total_seqs,
                            chunk_size=args.chunk_size,
                            use_fast_norm=args.use_fast_norm,
                            use_fast_rope=args.use_fast_rope,
                        ).loss

                        if isinstance(eval_loss, tuple):
                            eval_loss, _ = eval_loss

                        # Accumulate eval loss
                        eval_total_metric.add_(torch.stack([eval_loss, eval_acc]))
                        eval_total_steps += 1

                # Gather eval loss (reduce sum)
                eval_total_metric.div_(eval_total_steps)
                dist.reduce(eval_total_metric, 0)

                eval_epoch += 1

                if RANK == 0:
                    eval_loss, eval_acc = eval_total_metric.cpu().numpy()
                    mlflow.log_metrics(
                        metrics={"eval/loss": eval_loss, "eval/acc": eval_acc},
                        step=step,
                    )

                model_engine.train()

            ############ Save Checkpoint
            # Save model with lean state dict
            # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
            if (
                (step == train_total_steps)
                or (epoch + 1 == args.epochs)
                or (
                    args.save_strategy == "epoch"
                    and args.save_every
                    and ((epoch + 1) % args.save_every == 0)
                )
            ):
                dist.barrier()

                if model_engine.zero_optimization_stage() == 3:
                    state_dict = model_engine._zero3_consolidated_16bit_state_dict()
                elif RANK == 0:
                    state_dict = (
                        deepspeed.checkpoint.utils.clone_tensors_for_torch_save(
                            model_engine.module.state_dict()
                        )
                    )

                if RANK == 0:
                    save_path = os.path.join(args.save_path, f"ep_{epoch + 1}")

                    model_engine.module.save_pretrained(
                        save_path, state_dict=state_dict
                    )  # type: ignore

                    # Also save tokenizer from base model
                    save_tokenizer(args, save_path)

                    # Write metadata
                    save_openchat_metadata(args, epoch + 1, step, save_path)

    if RANK == 0:
        progress_bar.close()

        save_path = args.save_path

        model_engine.module.save_pretrained(
            save_path, state_dict=state_dict
        )  # type: ignore

        # Also save tokenizer from base model
        save_tokenizer(args, save_path)

        # Write metadata
        save_openchat_metadata(args, epoch + 1, step, save_path)

        mlflow.end_run()


if __name__ == "__main__":
    args = parse_args()
    args = TrainingArguments(**vars(args))
    with open(args.deepspeed_config) as f:
        deepspeed_config: dict = json.load(f)
    args.ds_zero_op = deepspeed_config.get("zero_optimization", {}).get("stage", 2)
    if deepspeed_config.get("zero_optimization", {}).get(
        "offload_optimizer", False
    ) or deepspeed_config.get("zero_optimization", {}).get("offload_param", False):
        args.ds_offload = True
        args.use_zero_one_opt = False
        args.use_qlora = False
    train(args)
