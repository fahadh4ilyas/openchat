"""Distributed ORPO training entry point (DeepSpeed).

ORPO (Odds Ratio Preference Optimization) combines SFT and preference
alignment without a reference model.  Supports both full fine-tuning and
LoRA/QLoRA.

base_lr=3e-4 (full FT), 1e-2 (LoRA).
"""

import argparse
import os
from functools import partial

import torch
import torch.distributed as dist

import tqdm
import mlflow

from transformers.integrations import HfDeepSpeedConfig

try:
    import deepspeed
except ImportError:
    raise ImportError("Please install deepspeed to train models.")

from ochat.config import MODEL_CONFIG_MAP
from ochat.training_utils._training_args import (
    BaseTrainingArguments,
    LoraTrainingArgsMixin,
    add_base_args,
    add_lora_args,
)
from ochat.training_utils._common import (
    combine_chosen_rejected_batch,
    mlflow_stopper_wrapper,
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
from ochat.training_orpo.utils import (
    orpo_batch_collate,
    _per_seq_response_tokens,
    orpo_loss,
)
from ochat.training_utils.multipack_dataloader import MultipackDistributedDataloader
from ochat.training_utils.numpy_dataset import NumpyDataset


class TrainingArguments(BaseTrainingArguments, LoraTrainingArgsMixin):
    """ORPO training arguments (base_lr=3e-4 full FT, 1e-2 LoRA)."""
    orpo_beta: float = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    add_base_args(parser, base_lr=3e-4)
    add_lora_args(parser)
    parser.add_argument("--orpo_beta", type=float, default=0.1,
                        help="ORPO temperature (λ in the paper, default 0.1)")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def create_distributed_dataloader(args, data: NumpyDataset):
    collate_fn = orpo_batch_collate
    if args.has_processor:
        tokenizer = load_tokenizer(args)
        collate_fn = partial(orpo_batch_collate, dataset_path=os.path.dirname(args.data_prefix),
                             processor=tokenizer)
    return MultipackDistributedDataloader(
        dataset=data,
        lengths=data["total_length"],
        numseqs=data["num_seqs"],
        batch_max_length=args.batch_max_len,
        collate_fn=collate_fn,
        seed=0,
    )


def _eval_loop(model_engine, eval_loader, args, eval_epoch):
    """Run one evaluation pass, return (sft_loss, orpo_loss, total_loss, next_epoch)."""
    eval_total_sft = torch.zeros((), dtype=torch.float32, device=args.device)
    eval_total_orpo = torch.zeros((), dtype=torch.float32, device=args.device)
    eval_total_loss = torch.zeros((), dtype=torch.float32, device=args.device)
    eval_total_steps = 0

    eval_loader.set_epoch(eval_epoch)
    with torch.inference_mode():
        for (chosen_t, rejected_t, batch_info), all_numseq, cur_numseq in eval_loader:
            chosen_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in chosen_t.items()}
            rejected_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in rejected_t.items()}

            combined_t, num_chosen = combine_chosen_rejected_batch(chosen_t, rejected_t)
            per_seq_logps = model_engine(
                **combined_t, **batch_info,
                num_seq=0,
                return_per_seq_logps=True,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).logits

            resp_tokens = _per_seq_response_tokens(combined_t)
            chosen_logp = per_seq_logps[:num_chosen] / resp_tokens[:num_chosen]
            rejected_logp = per_seq_logps[num_chosen:] / resp_tokens[num_chosen:]
            sft_loss = -chosen_logp.mean()
            orpo = orpo_loss(chosen_logp, rejected_logp, args.orpo_beta)
            eval_total_sft.add_(sft_loss)
            eval_total_orpo.add_(orpo)
            eval_total_loss.add_(sft_loss + orpo)
            eval_total_steps += 1

    eval_total_sft.div_(eval_total_steps)
    eval_total_orpo.div_(eval_total_steps)
    eval_total_loss.div_(eval_total_steps)
    dist.reduce(eval_total_sft, 0)
    dist.reduce(eval_total_orpo, 0)
    dist.reduce(eval_total_loss, 0)
    return eval_total_sft, eval_total_orpo, eval_total_loss, eval_epoch + 1


@mlflow_stopper_wrapper()
def train(args):
    from ochat.training_orpo.train_ring import train as train_ring

    if getattr(args, "use_ring", False):
        return train_ring(args)

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
    args.base_lr = 1e-2 if is_lora else 3e-4

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
        for (chosen_t, rejected_t, batch_info), all_numseq, cur_numseq in train_loader:
            step += 1
            if step > train_total_steps:
                break
            elif step <= latest_checkpoint:
                if RANK == 0:
                    progress_bar.update()
                continue

            # Move to device
            chosen_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in chosen_t.items()}
            rejected_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in rejected_t.items()}

            # Combine chosen + rejected → single forward + split per-seq log-probs
            combined_t, num_chosen = combine_chosen_rejected_batch(chosen_t, rejected_t)
            per_seq_logps = model_engine(
                **combined_t, **batch_info,
                num_seq=0,
                return_per_seq_logps=True,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).logits

            resp_tokens = _per_seq_response_tokens(combined_t)
            chosen_logp = per_seq_logps[:num_chosen] / resp_tokens[:num_chosen]
            rejected_logp = per_seq_logps[num_chosen:] / resp_tokens[num_chosen:]
            loss = -chosen_logp.mean() + orpo_loss(chosen_logp, rejected_logp, args.orpo_beta)

            model_engine.backward(loss)

            if model_engine.is_gradient_accumulation_boundary():
                lr_this_step = args.lr * lr_scheduler(step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_this_step

            model_engine.step()

            del combined_t, chosen_t, rejected_t
            if args.torch_empty_cache_steps is not None and step % args.torch_empty_cache_steps == 0:
                torch.cuda.empty_cache()

            if RANK == 0:
                mlflow.log_metrics(
                    metrics={
                        "train/loss": loss.item(),
                        "train/sft_loss": (-chosen_logp.mean()).item(),
                        "train/orpo_loss": (loss.item() - (-chosen_logp.mean()).item()),
                        "train/lr": lr_this_step,
                        "train/epoch": args.epochs * step / train_total_steps,
                    },
                    step=step,
                )
                progress_bar.update()

            if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0):
                dist.barrier()
                save_path = os.path.join(args.save_path, f"checkpoint_{step}")
                save_checkpoint(model_engine, args, save_path, epoch + 1, step)
                clean_checkpoint(args)

            if eval_loader is not None and args.eval_strategy == "step" and args.eval_every and (
                step % args.eval_every == 0
            ):
                model_engine.eval()
                eval_sft, eval_orpo, eval_loss, eval_epoch = _eval_loop(model_engine, eval_loader, args, eval_epoch)
                if RANK == 0:
                    mlflow.log_metrics(metrics={
                        "eval/loss": eval_loss.item(),
                        "eval/sft_loss": eval_sft.item(),
                        "eval/orpo_loss": eval_orpo.item(),
                    }, step=step)
                model_engine.train()

            if args.save_strategy == "step" and args.save_every and (step % args.save_every == 0):
                dist.barrier()
                save_path = os.path.join(args.save_path, f"st_{step}")
                save_checkpoint(model_engine, args, save_path, args.epochs * step / train_total_steps, step, save_tokenizer_too=True)

        if step > latest_checkpoint:
            if RANK == 0:
                mlflow.log_metrics(metrics={"batch_efficiency": train_loader.efficiency()}, step=step)

            if eval_loader is not None and (
                (step == train_total_steps) or (epoch + 1 == args.epochs) or
                (args.eval_strategy == "epoch" and args.eval_every and ((epoch + 1) % args.eval_every == 0))
            ):
                model_engine.eval()
                eval_sft, eval_orpo, eval_loss, eval_epoch = _eval_loop(model_engine, eval_loader, args, eval_epoch)
                if RANK == 0:
                    mlflow.log_metrics(metrics={
                        "eval/loss": eval_loss.item(),
                        "eval/sft_loss": eval_sft.item(),
                        "eval/orpo_loss": eval_orpo.item(),
                    }, step=step)
                model_engine.train()

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
