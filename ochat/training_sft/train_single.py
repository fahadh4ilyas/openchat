"""Single-GPU SFT/C-RLFT training entry point.

Handles full fine-tuning, LoRA, and QLoRA in one script.
base_lr: 3e-4 (full FT), auto-overridden to 1e-2 (LoRA).
"""

import argparse
import os
from functools import partial
from typing import Optional

from pydantic import Field

import torch

import tqdm
import mlflow

from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

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
    save_openchat_metadata,
    clean_checkpoint,
    save_tokenizer,
    load_tokenizer,
)
from ochat.training_utils.multipack_dataloader_single import MultipackDataloader
from ochat.training_utils.numpy_dataset import NumpyDataset


class TrainingArguments(BaseTrainingArguments, LoraTrainingArgsMixin):
    """Single-GPU SFT training arguments (base_lr=3e-4 full FT, 1e-2 LoRA)."""
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    add_base_args(parser, base_lr=3e-4)
    add_lora_args(parser)
    return parser.parse_args()


def create_distributed_dataloader(args: TrainingArguments, data: NumpyDataset):
    collate_fn = batch_to_tensor
    if args.has_processor:
        tokenizer = load_tokenizer(args)
        collate_fn = partial(batch_to_tensor, dataset_path=os.path.dirname(args.data_prefix), processor=tokenizer)
    return MultipackDataloader(
        dataset=data,
        lengths=data["total_length"],
        numseqs=data["num_seqs"],
        batch_max_length=args.batch_max_len,
        collate_fn=collate_fn,
        seed=0,
    )


def create_model(args: TrainingArguments):
    """Load model, optionally wrap with LoRA/QLoRA, return (model, optimizer)."""
    print(f"Loading model {args.model_type} from {args.model_path}...")

    model_path = get_latest_checkpoint(args) or args.model_path
    is_lora = args.use_lora or args.use_qlora

    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.quant_bits == 8,
            load_in_4bit=args.quant_bits == 4,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=args.quant_type_4bit,
            bnb_4bit_use_double_quant=args.use_double_quant_4bit,
        )

    model = MODEL_CONFIG_MAP[args.model_type].model_create_for_training(
        model_path if model_path == args.model_path else args.model_path,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False

    if is_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model)

        if model_path == args.model_path:
            lora_config = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules, exclude_modules=args.lora_exclude_modules,
                lora_dropout=args.lora_dropout, fan_in_fan_out=args.lora_fan_in_fan_out,
                use_rslora=args.lora_use_rslora, use_dora=args.lora_use_dora,
                use_qalora=args.lora_use_qalora, qalora_group_size=args.lora_qalora_group_size,
                bias=args.lora_bias, modules_to_save=args.lora_modules_to_save,
            )
            model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, model_path, is_trainable=True)

        model.enable_input_require_grads()

    model = model.to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        fused=True,
    )

    args.device = model.device
    return model, optimizer


def _run_eval(model, eval_loader, args, eval_epoch, step=None):
    model.eval()
    eval_total_metric = torch.zeros((2,), dtype=torch.float32, device=args.device)
    eval_total_steps = 0

    eval_loader.set_epoch(eval_epoch)
    with torch.inference_mode():
        for (batch_tensor, batch_info), num_seq in eval_loader:
            batch_tensor = {k: (v.to(args.device) if v is not None else None)
                            for k, v in batch_tensor.items()}
            eval_loss, eval_acc = model(
                **batch_tensor, **batch_info, num_seq=num_seq,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).loss
            if isinstance(eval_loss, tuple):
                eval_loss, _ = eval_loss
            eval_total_metric.add_(torch.stack([eval_loss, eval_acc]))
            eval_total_steps += 1

    eval_total_metric.div_(eval_total_steps)

    if step is not None:
        eval_loss, eval_acc = eval_total_metric.cpu().numpy()
        mlflow.log_metrics(metrics={"eval/loss": eval_loss, "eval/acc": eval_acc}, step=step)

    model.train()
    return eval_total_metric


@mlflow_stopper_wrapper(is_distributed=False)
def train(args: TrainingArguments):
    # Dataset
    train_dataset = create_dataset(args, "train")
    eval_dataset = create_dataset(args, "eval")

    if train_dataset is None:
        raise RuntimeError("Training data not found.")

    # Load model type
    args.model_type = train_dataset.metadata["model_type"]
    args.has_processor = MODEL_CONFIG_MAP[args.model_type].model_has_processor

    # Adjust base_lr for LoRA (adapters converge faster)
    if args.use_lora or args.use_qlora:
        args.base_lr = 1e-2

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
        args.base_lr, args.lr, args.batch_max_len, train_dataset,
        MODEL_CONFIG_MAP[args.model_type].base_lr_scale,
    )

    # Logger
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
    model, optimizer = create_model(args)

    # LR Scheduler
    lr_scheduler = create_lr_scheduler(args, train_total_steps)

    # Progress bar
    progress_bar = tqdm.tqdm(total=train_total_steps)

    # Training Loop
    step = 0
    latest_checkpoint = int((get_latest_checkpoint(args) or "_0").split("_")[-1])
    lr_this_step = None
    model.train()
    eval_epoch = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        train_loader.set_epoch(epoch)
        for (batch_tensor, batch_info), num_seq in train_loader:
            step += 1
            if step > train_total_steps:  # At most train_total_steps
                break
            elif step <= latest_checkpoint:
                progress_bar.update()
                continue

            optimizer.zero_grad()

            # To device
            batch_tensor = {
                k: (v.to(args.device) if v is not None else None)
                for k, v in batch_tensor.items()
            }

            # Update
            loss, acc = model(
                **batch_tensor,
                **batch_info,
                num_seq=num_seq,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).loss

            if isinstance(loss, tuple):
                loss, _ = loss

            loss.backward()

            # Set LR
            lr_this_step = args.lr * lr_scheduler(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_this_step

            optimizer.step()

            del batch_tensor
            if args.torch_empty_cache_steps is not None and step % args.torch_empty_cache_steps == 0:
                torch.cuda.empty_cache()

            # Logging
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
                save_path = os.path.join(args.save_path, f"checkpoint_{step}")

                model.save_pretrained(save_path)  # type: ignore

                # Write metadata
                save_openchat_metadata(args, epoch + 1, step, save_path)

                clean_checkpoint(args)

            if eval_loader is not None and (
                args.eval_strategy == "step"
                and args.eval_every
                and (step % args.eval_every == 0)
            ):
                _run_eval(model, eval_loader, args, eval_epoch, step=step)
                eval_epoch += 1

            if (
                args.save_strategy == "step"
                and args.save_every
                and (step % args.save_every == 0)
            ):
                save_path = os.path.join(args.save_path, f"st_{step}")

                model.save_pretrained(save_path)  # type: ignore

                # Also save tokenizer from base model
                save_tokenizer(args, save_path)

                # Write metadata
                save_openchat_metadata(
                    args, args.epochs * step / train_total_steps, step, save_path
                )

        if step > latest_checkpoint:
            # Log batch efficiency
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
                _run_eval(model, eval_loader, args, eval_epoch, step=step)
                eval_epoch += 1

            ############ Save Checkpoint
            if (args.save_strategy == "epoch"
                and args.save_every
                and ((epoch + 1) % args.save_every == 0)
            ):
                save_path = os.path.join(args.save_path, f"ep_{epoch + 1}")

                model.save_pretrained(save_path)  # type: ignore

                # Also save tokenizer from base model
                save_tokenizer(args, save_path)

                # Write metadata
                save_openchat_metadata(args, epoch + 1, step, save_path)

    progress_bar.close()

    save_path = args.save_path

    model.save_pretrained(save_path)  # type: ignore

    # Also save tokenizer from base model
    save_tokenizer(args, save_path)

    # Write metadata
    save_openchat_metadata(args, epoch + 1, step, save_path)

    mlflow.end_run()


if __name__ == "__main__":
    args = parse_args()
    args = TrainingArguments(**vars(args))
    train(args)
