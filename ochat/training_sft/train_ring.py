"""Ring-attention distributed SFT/C-RLFT training.

Handles full fine-tuning, LoRA, and QLoRA in one script.
Uses multipack_dataloader_ring for ring-attention-based batching.

base_lr=3e-4 (full FT), 1e-2 (LoRA).
"""

import argparse
import os
from functools import partial

import torch
import torch.distributed as dist

import tqdm
import mlflow

from ochat.config import MODEL_CONFIG_MAP
from ochat.training_utils._training_args import (
    BaseTrainingArguments,
    LoraTrainingArgsMixin,
    add_base_args,
    add_lora_args,
)
from ochat.training_utils._common import (
    mlflow_stopper_wrapper,
    batch_to_tensor,
    get_latest_checkpoint,
    create_dataset,
    calculate_auto_lr,
    create_lr_scheduler,
    clean_checkpoint,
    load_tokenizer,
)
from ochat.training_utils.base_train import (
    create_model_and_engine,
    save_checkpoint,
    setup_mlflow,
    _parse_ds_config,
)
from ochat.training_utils.multipack_dataloader_ring import MultipackDistributedDataloader
from ochat.training_utils.numpy_dataset import NumpyDataset

from transformers.integrations import HfDeepSpeedConfig

try:
    import deepspeed
except ImportError:
    raise ImportError("Please install deepspeed to train models.")


class TrainingArguments(BaseTrainingArguments, LoraTrainingArgsMixin):
    """Ring-attention training arguments. LoRA fields are ignored when --use-lora is not set."""
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    add_base_args(parser, base_lr=3e-4)
    add_lora_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def create_distributed_dataloader(args, data):
    collate_fn = batch_to_tensor
    if args.has_processor:
        tokenizer = load_tokenizer(args)
        collate_fn = partial(batch_to_tensor, dataset_path=os.path.dirname(args.data_prefix), processor=tokenizer)
    return MultipackDistributedDataloader(
        dataset=data,
        lengths=data["total_length"],
        batch_max_length=args.batch_max_len,
        collate_fn=collate_fn,
        seed=0,
    )


def _run_eval(model_engine, eval_loader, args, eval_epoch, step=None):
    model_engine.eval()
    eval_total_metric = torch.zeros((2,), dtype=torch.float32, device=args.device)
    eval_total_steps = 0

    eval_loader.set_epoch(eval_epoch)
    with torch.inference_mode():
        for (batch_tensor, batch_info), total_seqs in eval_loader:
            batch_tensor = {k: (v.to(args.device) if v is not None else None)
                            for k, v in batch_tensor.items()}
            eval_loss, eval_acc = model_engine(
                **batch_tensor, **batch_info, total_seqs=total_seqs,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).loss
            if isinstance(eval_loss, tuple):
                eval_loss, _ = eval_loss
            eval_total_metric.add_(torch.stack([eval_loss, eval_acc]))
            eval_total_steps += 1

    eval_total_metric.div_(eval_total_steps)
    dist.reduce(eval_total_metric, 0)

    if dist.get_rank() == 0 and step is not None:
        eval_loss, eval_acc = eval_total_metric.cpu().numpy()
        mlflow.log_metrics(metrics={"eval/loss": eval_loss, "eval/acc": eval_acc}, step=step)

    model_engine.train()
    return eval_total_metric


@mlflow_stopper_wrapper()
def train(args):
    deepspeed.init_distributed(dist_backend="nccl")
    dsconfig = HfDeepSpeedConfig(args.deepspeed_config)
    RANK = dist.get_rank()

    train_dataset = create_dataset(args, "train")
    eval_dataset = create_dataset(args, "eval")

    if train_dataset is None:
        raise RuntimeError("Training data not found.")

    args.model_type = train_dataset.metadata["model_type"]
    args.has_processor = MODEL_CONFIG_MAP[args.model_type].model_has_processor

    is_lora = args.use_lora or args.use_qlora
    args.base_lr = 1e-2 if is_lora else args.base_lr

    train_loader = create_distributed_dataloader(args, train_dataset)
    if args.max_steps > 0:
        args.epochs = -(-args.max_steps // train_loader.num_batches())
        train_total_steps = args.max_steps
    else:
        train_total_steps = args.epochs * train_loader.num_batches()

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = create_distributed_dataloader(args, eval_dataset)

    args.lr = calculate_auto_lr(args.base_lr, args.lr, args.batch_max_len, args.model_type, train_dataset)

    setup_mlflow(args, train_total_steps, RANK)

    model_engine, optimizer = create_model_and_engine(args, args.base_lr)
    lr_scheduler = create_lr_scheduler(args, train_total_steps)

    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_total_steps)

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
            if step > train_total_steps:
                break
            elif step <= latest_checkpoint:
                if RANK == 0:
                    progress_bar.update()
                continue

            batch_tensor = {k: (v.to(args.device) if v is not None else None)
                            for k, v in batch_tensor.items()}

            loss, acc = model_engine(
                **batch_tensor, **batch_info, total_seqs=total_seqs,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).loss

            if isinstance(loss, tuple):
                loss, _ = loss

            model_engine.backward(loss)

            if model_engine.is_gradient_accumulation_boundary():
                lr_this_step = args.lr * lr_scheduler(step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_this_step

            model_engine.step()

            dist.reduce(loss, 0)
            dist.reduce(acc, 0)

            del batch_tensor
            if args.torch_empty_cache_steps is not None and step % args.torch_empty_cache_steps == 0:
                torch.cuda.empty_cache()

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
                progress_bar.update()

            if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0):
                dist.barrier()
                save_checkpoint(model_engine, args, os.path.join(args.save_path, f"checkpoint_{step}"),
                                epoch + 1, step)
                clean_checkpoint(args)

            if eval_loader is not None and args.eval_strategy == "step" and args.eval_every and (
                step % args.eval_every == 0
            ):
                _run_eval(model_engine, eval_loader, args, eval_epoch, step=step)
                eval_epoch += 1

            if args.save_strategy == "step" and args.save_every and (step % args.save_every == 0):
                dist.barrier()
                save_checkpoint(model_engine, args, os.path.join(args.save_path, f"st_{step}"),
                                args.epochs * step / train_total_steps, step, save_tokenizer_too=True)

        if step > latest_checkpoint:
            if RANK == 0:
                mlflow.log_metrics(metrics={"batch_efficiency": train_loader.efficiency()}, step=step)

            if eval_loader is not None and (
                (step == train_total_steps) or (epoch + 1 == args.epochs) or
                (args.eval_strategy == "epoch" and args.eval_every and ((epoch + 1) % args.eval_every == 0))
            ):
                _run_eval(model_engine, eval_loader, args, eval_epoch, step=step)
                eval_epoch += 1

            if args.save_strategy == "epoch" and args.save_every and ((epoch + 1) % args.save_every == 0):
                dist.barrier()
                save_path = os.path.join(args.save_path, f"ep_{epoch + 1}")
                save_checkpoint(model_engine, args, save_path, epoch + 1, step, save_tokenizer_too=True)

    if RANK == 0:
        progress_bar.close()
    save_checkpoint(model_engine, args, args.save_path, epoch + 1, step, save_tokenizer_too=True)
    if RANK == 0:
        mlflow.end_run()


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    _parse_ds_config(args_dict)
    args = TrainingArguments(**args_dict)
    train(args)
