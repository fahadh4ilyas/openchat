import argparse
import os
import math
import json
import shutil
from pathlib import Path
from functools import partial
from typing import Optional

from pydantic import BaseModel, Field, validator

import torch
import torch.distributed as dist

import tqdm
import mlflow
import numpy as np

from ochat.config import MODEL_CONFIG_MAP
from ochat.training_deepspeed.multipack_dataloader_ring import MultipackDistributedDataloader
from ochat.training_deepspeed.numpy_dataset import NumpyDataset

from transformers.integrations import HfDeepSpeedConfig

from flash_attn.losses.cross_entropy import CrossEntropyLoss

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
    checkpoint_every: int = Field(0, ge=0)
    max_checkpoint: int = Field(1, gt=0)
    batch_max_len: int = Field(81920)
    epochs: int = Field(5)
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
    deepspeed_config: str = Field(...)
    deepspeed_mpi: bool = Field(False)
    ds_zero_op: int = Field(0)
    device: Optional[str] = Field(None)

    @validator('batch_max_len')
    def val_batch_size(cls, v: int) -> int:

        if v%2048 != 0:
            raise ValueError('`batch_max_len` must be multiple of 2048')
        
        return v
    
    @validator('use_zero_one_opt')
    def val_opt(cls, v: bool) -> bool:

        if v:
            raise ValueError('`use_zero_one_opt` must be False for offloading')
        
        return v


PAD_ID     = 0
LABEL_PAD_ID = -100
BATCH_KEYS = {
    "seqlens": torch.long,
    "nz_input_ids": torch.long,
    "nz_position_ids": torch.long,
    "nz_shifted_label_ids": torch.long,

    "nz_shifted_loss_weights": torch.bfloat16
}

MODEL_LR = ['mistral', 'qwen2', 'mixtral']


def _find_multiple(a, b):
    return (-(a // -b)) * b

def parse_args():
    parser_base = argparse.ArgumentParser(add_help=False)
    # parser_lora_confirm = argparse.ArgumentParser(add_help=False)
    # parser_lora = argparse.ArgumentParser(add_help=False)
    # Distributed
    parser_base.add_argument("--local_rank",            type=int, required=True)

    # Model type and data
    parser_base.add_argument("--model_path",            type=str, required=True)
    parser_base.add_argument("--data_prefix",           type=str, required=True)
    parser_base.add_argument("--save_path",             type=str, required=True)
    parser_base.add_argument("--save_every",            type=int, default=None)
    parser_base.add_argument("--checkpoint_every",      type=int, default=0)
    parser_base.add_argument("--max_checkpoint",        type=int, default=1)

    # Hyperparameters
    parser_base.add_argument("--batch_max_len",         type=int, default=81920)
    parser_base.add_argument("--epochs",                type=int,   default=5)

    # Set lr to None to automatically estimate from LLaMA pretraining parameters (e.g. lr ~ sqrt(batch_size))
    parser_base.add_argument("--use_zero_one_opt",      action='store_true')
    parser_base.add_argument("--base_lr",               type=float, default=3e-4)
    parser_base.add_argument("--lr",                    type=float, default=None)
    parser_base.add_argument("--lr_min_ratio",          type=float, default=0.1)
    parser_base.add_argument("--lr_warmup_ratio",       type=float,   default=0.05)
    parser_base.add_argument("--lr_warmup_step",        type=int,   default=0)

    parser_base.add_argument("--weight_decay",          type=float, default=0.1)

    parser_base.add_argument("--beta1",                 type=float, default=0.9)
    parser_base.add_argument("--beta2",                 type=float, default=0.95)
    parser_base.add_argument("--eps",                   type=float, default=1e-5)

    # MLFLOW
    parser_base.add_argument("--tracking_uri",          type=str, default=None)
    parser_base.add_argument("--mlflow_username",       type=str, default=None)
    parser_base.add_argument("--mlflow_password",       type=str, default=None)
    parser_base.add_argument("--experiment_name",       type=str, required=True)
    parser_base.add_argument("--run_name",              type=str, required=True)

    # LORA
    # parser_lora_confirm.add_argument("--use_lora",      action='store_true')
    # parser_lora.add_argument("--lora_alpha",            type=int, default=32)
    # parser_lora.add_argument("--lora_r",                type=int, default=32)
    # parser_lora.add_argument("--lora_dropout",          type=float, default=0.05)
    # parser_lora.add_argument("--lora_target_modules",   type=str, nargs="*", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    # parser_lora.add_argument("--lora_bias",             type=str, default="none")
    # parser_lora.add_argument("--modules_to_save",       type=str, nargs="*", default=None)

    # DeepSpeed parameters
    parser_base = deepspeed.add_config_arguments(parser_base)

    # Group parser
    # parser_group = argparse.ArgumentParser(parents=[parser_base, parser_lora_confirm, parser_lora])

    # Parse known args
    # parser_group.parse_args()
    args_base, _ = parser_base.parse_known_args()
    # args_lora_confirm, _ = parser_lora_confirm.parse_known_args()
    # args_lora, _ = parser_lora.parse_known_args()
    return args_base # , args_lora_confirm, args_lora


def create_dataset(args: TrainingArguments, split_name):
    # Load data
    filename = f"{args.data_prefix}.{split_name}.parquet"
    if not os.path.isfile(filename):
        print (f"Skipping loading {split_name}")
        return None

    print(f"Loading {split_name} data from {filename}...")
    return NumpyDataset(filename)


def batch_to_tensor(batch):
    # Concat batches
    batch = {k: np.concatenate(batch[k], axis=0) for k in BATCH_KEYS.keys()}

    # Pad an unused item to reach multiple of 64, for faster GEMM
    total_seqlen = batch["nz_input_ids"].size
    pad_len      = _find_multiple(total_seqlen, 64) - total_seqlen

    if pad_len > 0:
        assert pad_len < 64

        # total length
        padding_specs = {
            "seqlens": (1, pad_len),

            "nz_input_ids": (pad_len, PAD_ID),
            "nz_position_ids": (pad_len, 0),
            "nz_shifted_label_ids": (pad_len, LABEL_PAD_ID),
            "nz_shifted_loss_weights": (pad_len, 0),
        }
        for k, pad_spec in padding_specs.items():
            batch[k] = np.concatenate((batch[k], np.full(*pad_spec, dtype=batch[k].dtype)), axis=0)

    # to tensor
    batch_tensor = {}
    for k, dtype in BATCH_KEYS.items():
        batch_tensor[k] = torch.from_numpy(batch[k]).to(dtype)

    # cu seqlens
    batch_tensor["cu_seqlens"] = torch.nn.functional.pad(batch_tensor["seqlens"].cumsum(-1, dtype=torch.int32), (1, 0))
    # batch info
    batch_info = {"max_seqlen": torch.max(batch_tensor["seqlens"]).item()}

    # inputs
    del batch_tensor["seqlens"]
    return batch_tensor, batch_info


def create_distributed_dataloader(args: TrainingArguments, data):
    # Multipack dataloader
    return MultipackDistributedDataloader(
        dataset=data,
        lengths=data["total_length"],

        batch_max_length=args.batch_max_len,
        collate_fn=batch_to_tensor,

        seed=0
    )

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
    model = MODEL_CONFIG_MAP[args.model_type].model_create_for_training(model_path, low_cpu_mem_usage=args.ds_zero_op != 3)
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Optimizer
    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(model.parameters(),
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


def create_lr_scheduler(args: TrainingArguments, train_total_steps):
    lr_scheduler = partial(
        cosine_schedule_with_warmup_lr_lambda,

        num_warmup_steps=args.lr_warmup_step or round(args.lr_warmup_ratio * train_total_steps),
        num_training_steps=train_total_steps,
        min_ratio=args.lr_min_ratio
    )

    return lr_scheduler


def save_tokenizer(args: TrainingArguments, save_path):
    MODEL_CONFIG_MAP[args.model_type].model_tokenizer_create(args.model_path).save_pretrained(save_path)


def save_openchat_metadata(args: TrainingArguments, epoch, latest_step, save_path):
    metadata = vars(args)
    metadata["epoch"] = epoch
    metadata["latest_step"] = latest_step

    with open(os.path.join(save_path, "openchat.json"), "w") as f:
        json.dump(metadata, f, default=lambda o: "<non-serializable>")


def calculate_auto_lr(base_lr, lr, batch_max_len, model_type, train_dataset):
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

    print(f"Use automatic learning rate {lr} (estimated from supervised ratio {supervised_ratio} effective batch size {supervised_tokens})")
    return lr


def train(args: TrainingArguments):
    deepspeed.init_distributed(dist_backend="nccl")
    dsconfig = HfDeepSpeedConfig(args.deepspeed_config)
    RANK = dist.get_rank()

    # Dataset
    train_dataset = create_dataset(args, "train")
    eval_dataset  = create_dataset(args, "eval")

    if train_dataset is None:
        raise RuntimeError("Training data not found.")

    # Load model type
    args.model_type = train_dataset.metadata["model_type"]

    # Data Loader
    train_loader      = create_distributed_dataloader(args, train_dataset)
    train_total_steps = args.epochs * train_loader.num_batches()

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = create_distributed_dataloader(args, eval_dataset)

    # Hyperparams
    args.lr = calculate_auto_lr(args.base_lr, args.lr, args.batch_max_len, args.model_type, train_dataset)

    # Logger
    if RANK == 0:

        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        if args.mlflow_username:
            os.environ['MLFLOW_TRACKING_USERNAME'] = args.mlflow_username
        if args.mlflow_password:
            os.environ['MLFLOW_TRACKING_PASSWORD'] = args.mlflow_username
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run(run_name=args.run_name)
        metadata = vars(args).copy()
        metadata.pop('local_rank', None)
        metadata.pop('device', None)
        metadata['steps'] = train_total_steps
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
    latest_checkpoint = int((get_latest_checkpoint(args) or '_0').split('_')[-1])
    lr_this_step = None
    loss_func = CrossEntropyLoss(inplace_backward=True)
    for epoch in range(args.epochs):
        print (f"[rank {RANK}]: Epoch {epoch}")

        ############ Train Epoch
        model_engine.train()

        train_loader.set_epoch(epoch)
        for batch_tensor, batch_info in train_loader:
            step += 1
            if step > train_total_steps:  # At most train_total_steps
                break
            elif step <= latest_checkpoint:
                if RANK == 0:
                    progress_bar.update()
                continue

            # To device
            batch_tensor = {k: (v.to(args.device) if v is not None else None) for k, v in batch_tensor.items()}

            # Update
            output = model_engine(**batch_tensor, **batch_info)
            acc = output.loss
            logits = output.logits
            loss = loss_func(logits, batch_tensor['nz_shifted_label_ids'])

            model_engine.backward(loss)

            if model_engine.is_gradient_accumulation_boundary():
                # Set LR
                lr_this_step = args.lr * lr_scheduler(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

            model_engine.step()

            dist.reduce(loss, 0, dist.ReduceOp.AVG)
            dist.reduce(acc, 0, dist.ReduceOp.AVG)

            # Logging
            if RANK == 0:
                mlflow.log_metrics(metrics={
                    "train/loss": loss.item(),
                    "train/acc":  acc.item(),
                    "train/lr": lr_this_step,
                    "train/epoch": args.epochs * step / train_total_steps
                }, step=step)
                progress_bar.update()  # type: ignore

            if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0):
                dist.barrier()

                if model_engine.zero_optimization_stage() == 3:
                    state_dict = model_engine._zero3_consolidated_16bit_state_dict()
                elif RANK == 0:
                    state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_engine.module.state_dict())

                if RANK == 0:
                    save_path = os.path.join(args.save_path, f"checkpoint_{step}")

                    model_engine.module.save_pretrained(save_path,
                                                        state_dict=state_dict)  # type: ignore

                    # Write metadata
                    save_openchat_metadata(args, epoch + 1, step, save_path)

                    clean_checkpoint(args)

        if step > latest_checkpoint:

            # Log batch efficiency
            if RANK == 0:
                mlflow.log_metrics(metrics={"batch_efficiency": train_loader.efficiency()}, step=step)

            ############ Eval Epoch
            if eval_loader is not None:
                model_engine.eval()

                eval_total_metric = torch.zeros((2, ), dtype=torch.float32, device=args.device)
                eval_total_steps = 0

                eval_loader.set_epoch(epoch)
                with torch.inference_mode():
                    for batch_tensor, batch_info in eval_loader:
                        # To device
                        batch_tensor = {k: (v.to(args.device) if v is not None else None) for k, v in batch_tensor.items()}

                        # Eval
                        output = model_engine(**batch_tensor, **batch_info)
                        eval_acc = output.loss
                        eval_logits = output.logits
                        eval_loss = loss_func(eval_logits, batch_tensor['nz_shifted_label_ids'])
                        
                        # Accumulate eval loss
                        eval_total_metric.add_(torch.stack([eval_loss, eval_acc]))
                        eval_total_steps += 1

                # Gather eval loss (reduce sum)
                eval_total_metric.div_(eval_total_steps)
                dist.reduce(eval_total_metric, 0)

                if RANK == 0:
                    eval_loss, eval_acc = eval_total_metric.cpu().numpy()
                    mlflow.log_metrics(metrics={"eval/loss": eval_loss, "eval/acc": eval_acc}, step=step)

            ############ Save Checkpoint
            # Save model with lean state dict
            # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
            if (epoch + 1 == args.epochs) or (args.save_every and ((epoch + 1) % args.save_every == 0)):
                dist.barrier()

                if model_engine.zero_optimization_stage() == 3:
                    state_dict = model_engine._zero3_consolidated_16bit_state_dict()
                elif RANK == 0:
                    state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_engine.module.state_dict())

                if RANK == 0:
                    save_path = os.path.join(args.save_path, f"ep_{epoch + 1}")

                    model_engine.module.save_pretrained(save_path,
                                                        state_dict=state_dict)  # type: ignore

                    # Also save tokenizer from base model
                    save_tokenizer(args, save_path)

                    # Write metadata
                    save_openchat_metadata(args, epoch + 1, step, save_path)
    
    if RANK == 0:
        mlflow.end_run()


if __name__ == "__main__":
    # args, args_lora_confirm, args_lora = parse_args()
    args = parse_args()
    args = TrainingArguments(**vars(args))
    with open(args.deepspeed_config) as f:
        deepspeed_config = json.load(f)
    args.ds_zero_op = deepspeed_config.get('zero_optimization', {}).get('stage', 2)
    args.use_zero_one_opt = False
    train(args)
