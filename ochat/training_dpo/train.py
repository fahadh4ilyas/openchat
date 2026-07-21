"""Distributed LoRA DPO training entry point (DeepSpeed).

Only LoRA/QLoRA: the frozen base model serves as the reference model.
Routes to train_ring if --use_ring is set.

base_lr=1e-2 (LoRA adapters converge faster than full fine-tuning).
"""

import argparse
import os
import json
from functools import partial
from typing import Optional, Union, Literal, List

import torch
import torch.distributed as dist

import tqdm
import mlflow

from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

from ochat.config import MODEL_CONFIG_MAP
from ochat.training_utils._training_args import (
    BaseTrainingArguments,
    LoraTrainingArgsMixin,
    add_base_args,
    add_lora_args,
)
from ochat.training_dpo.utils import (
    mlflow_stopper_wrapper,
    dpo_batch_collate,
    _combine_chosen_rejected_batch,
    dpo_loss,
    check_ref_logps_precomputed,
    get_latest_checkpoint,
    create_dataset,
    calculate_auto_lr,
    create_lr_scheduler,
    save_openchat_metadata,
    clean_checkpoint,
    save_tokenizer,
    load_tokenizer,
)
from ochat.training_utils.multipack_dataloader import MultipackDistributedDataloader
from ochat.training_utils.numpy_dataset import NumpyDataset

from transformers.integrations import HfDeepSpeedConfig
from transformers import BitsAndBytesConfig

try:
    import deepspeed
except ImportError:
    raise ImportError("Please install deepspeed to train models.")


class TrainingArguments(BaseTrainingArguments, LoraTrainingArgsMixin):
    """DPO training arguments (LoRA only, base_lr=1e-2)."""
    base_lr: float = 1e-2
    dpo_beta: float = 0.1
    use_ring: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ring", action="store_true")
    add_base_args(parser, base_lr=1e-2)
    add_lora_args(parser)
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO temperature parameter")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def create_distributed_dataloader(args, data: NumpyDataset):
    collate_fn = dpo_batch_collate
    if args.has_processor:
        tokenizer = load_tokenizer(args)
        collate_fn = partial(dpo_batch_collate, dataset_path=os.path.dirname(args.data_prefix), processor=tokenizer)
    return MultipackDistributedDataloader(
        dataset=data,
        lengths=data["total_length"],
        numseqs=data["num_seqs"],
        batch_max_length=args.batch_max_len,
        collate_fn=collate_fn,
        seed=0,
    )


def create_model(args):
    print(f"Loading model {args.model_type} from {args.model_path}...")

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

    model = MODEL_CONFIG_MAP[args.model_type].model_create_for_training(
        model_path if model_path == args.model_path else args.model_path,
        low_cpu_mem_usage=args.ds_zero_op != 3,
        quantization_config=quantization_config,
    ).to(args.local_rank)

    model.config.use_cache = False
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    if model_path == args.model_path:
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules, lora_dropout=args.lora_dropout,
            bias=args.lora_bias, modules_to_save=args.modules_to_save,
        )
        model = get_peft_model(model, lora_config)
    else:
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)

    if not args.ds_offload:
        model = model.to(args.local_rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()

    if args.ds_offload:
        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2), eps=args.eps,
        )
    elif args.use_zero_one_opt:
        with open(args.deepspeed_config) as f:
            ds_config: dict = json.load(f)
        ds_config["optimizer"] = {
            "type": "ZeroOneAdam",
            "params": {"lr": args.lr, "weight_decay": args.weight_decay, "betas": [args.beta1, args.beta2], "eps": args.eps},
        }
        ds_config.pop("zero_optimization", None)
        args.deepspeed_config = ds_config
        optimizer = None
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2), eps=args.eps, fused=True,
        )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(), optimizer=optimizer
    )
    args.device = model_engine.device
    return model_engine, optimizer


def _forward_and_logp(model_engine, batch_tensor, batch_info, args, all_numseq):
    """Run model forward and return per-example sum of log-probs.

    With loss weights = 1 for response tokens and 0 for prompt:
        sum(log_probs_response) = -loss
    """
    loss, acc = model_engine(
        **batch_tensor,
        **batch_info,
        num_seq=all_numseq,
        chunk_size=args.chunk_size,
        use_fast_norm=args.use_fast_norm,
        use_fast_rope=args.use_fast_rope,
    ).loss

    if isinstance(loss, tuple):
        loss, _ = loss

    return -loss


def _save_state_dict(model_engine, args):
    if model_engine.zero_optimization_stage() == 3:
        return model_engine._zero3_consolidated_16bit_state_dict()
    elif dist.get_rank() == 0:
        return deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_engine.module.state_dict())
    return None


@mlflow_stopper_wrapper()
def train(args):
    from ochat.training_dpo.train_ring import train as train_ring

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

    ref_logps_precomputed = check_ref_logps_precomputed(train_dataset)
    if not ref_logps_precomputed and RANK == 0:
        print("Reference log-probs not precomputed — will compute online (disabling LoRA adapters)")

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

    model_engine, optimizer = create_model(args)
    lr_scheduler = create_lr_scheduler(args, train_total_steps)

    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_total_steps)

    step = 0
    latest_checkpoint = int((get_latest_checkpoint(args) or "_0").split("_")[-1])
    lr_this_step = None
    model_engine.train()
    eval_epoch = 0
    state_dict = None

    for epoch in range(args.epochs):
        print(f"[rank {RANK}]: Epoch {epoch}")

        train_loader.set_epoch(epoch)
        for (chosen_t, rejected_t, chosen_ref, rejected_ref, batch_info), all_numseq, cur_numseq in train_loader:
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

            # Reference log-probs: precomputed in dataset, or online via frozen base model
            if ref_logps_precomputed:
                chosen_ref = chosen_ref.to(args.device)
                rejected_ref = rejected_ref.to(args.device)
            else:
                model_engine.module.disable_adapter_layers()
                with torch.no_grad():
                    chosen_ref = _forward_and_logp(model_engine, chosen_t, batch_info, args, all_numseq)
                    rejected_ref = _forward_and_logp(model_engine, rejected_t, batch_info, args, all_numseq)
                model_engine.module.enable_adapter_layers()

            # Combine chosen + rejected → single forward + split per-seq log-probs
            combined_t, num_chosen = _combine_chosen_rejected_batch(chosen_t, rejected_t)
            per_seq_logps = model_engine(
                **combined_t, **batch_info,
                num_seq=0,
                return_per_seq_logps=True,
                chunk_size=args.chunk_size,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).logits

            chosen_logp = per_seq_logps[:num_chosen]
            rejected_logp = per_seq_logps[num_chosen:]
            loss = dpo_loss(chosen_logp, rejected_logp, chosen_ref, rejected_ref, args.dpo_beta)

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
                        "train/lr": lr_this_step,
                        "train/epoch": args.epochs * step / train_total_steps,
                    },
                    step=step,
                )
                progress_bar.update()

            if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0):
                dist.barrier()
                state_dict = _save_state_dict(model_engine, args)
                if RANK == 0 and state_dict is not None:
                    save_path = os.path.join(args.save_path, f"checkpoint_{step}")
                    model_engine.module.save_pretrained(save_path, state_dict=state_dict)
                    save_openchat_metadata(args, epoch + 1, step, save_path)
                    clean_checkpoint(args)

            if eval_loader is not None and args.eval_strategy == "step" and args.eval_every and (step % args.eval_every == 0):
                model_engine.eval()
                eval_total_loss = torch.zeros((), dtype=torch.float32, device=args.device)
                eval_total_steps = 0

                eval_loader.set_epoch(eval_epoch)
                with torch.inference_mode():
                    for (chosen_t, rejected_t, chosen_ref, rejected_ref, batch_info), all_numseq, cur_numseq in eval_loader:
                        chosen_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in chosen_t.items()}
                        rejected_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in rejected_t.items()}

                        if ref_logps_precomputed:
                            chosen_ref = chosen_ref.to(args.device)
                            rejected_ref = rejected_ref.to(args.device)
                        else:
                            model_engine.module.disable_adapter_layers()
                            chosen_ref = _forward_and_logp(model_engine, chosen_t, batch_info, args, all_numseq)
                            rejected_ref = _forward_and_logp(model_engine, rejected_t, batch_info, args, all_numseq)
                            model_engine.module.enable_adapter_layers()

                        combined_t, num_chosen = _combine_chosen_rejected_batch(chosen_t, rejected_t)
                        per_seq_logps = model_engine(
                            **combined_t, **batch_info,
                            num_seq=0,
                            return_per_seq_logps=True,
                            chunk_size=args.chunk_size,
                            use_fast_norm=args.use_fast_norm,
                            use_fast_rope=args.use_fast_rope,
                        ).logits
                        chosen_logp = per_seq_logps[:num_chosen]
                        rejected_logp = per_seq_logps[num_chosen:]
                        eval_loss = dpo_loss(chosen_logp, rejected_logp, chosen_ref, rejected_ref, args.dpo_beta)

                        eval_total_loss.add_(eval_loss)
                        eval_total_steps += 1

                eval_total_loss.div_(eval_total_steps)
                dist.reduce(eval_total_loss, 0)
                eval_epoch += 1

                if RANK == 0:
                    mlflow.log_metrics(metrics={"eval/loss": eval_total_loss.item()}, step=step)
                model_engine.train()

            if args.save_strategy == "step" and args.save_every and (step % args.save_every == 0):
                dist.barrier()
                state_dict = _save_state_dict(model_engine, args)
                if RANK == 0 and state_dict is not None:
                    save_path = os.path.join(args.save_path, f"st_{step}")
                    model_engine.module.save_pretrained(save_path, state_dict=state_dict)
                    save_tokenizer(args, save_path)
                    save_openchat_metadata(args, args.epochs * step / train_total_steps, step, save_path)

        if step > latest_checkpoint:
            if RANK == 0:
                mlflow.log_metrics(metrics={"batch_efficiency": train_loader.efficiency()}, step=step)

            if eval_loader is not None and (
                (step == train_total_steps) or (epoch + 1 == args.epochs) or
                (args.eval_strategy == "epoch" and args.eval_every and ((epoch + 1) % args.eval_every == 0))
            ):
                model_engine.eval()
                eval_total_loss = torch.zeros((), dtype=torch.float32, device=args.device)
                eval_total_steps = 0

                eval_loader.set_epoch(eval_epoch)
                with torch.inference_mode():
                    for (chosen_t, rejected_t, chosen_ref, rejected_ref, batch_info), all_numseq, cur_numseq in eval_loader:
                        chosen_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in chosen_t.items()}
                        rejected_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in rejected_t.items()}

                        if ref_logps_precomputed:
                            chosen_ref = chosen_ref.to(args.device)
                            rejected_ref = rejected_ref.to(args.device)
                        else:
                            model_engine.module.disable_adapter_layers()
                            chosen_ref = _forward_and_logp(model_engine, chosen_t, batch_info, args, all_numseq)
                            rejected_ref = _forward_and_logp(model_engine, rejected_t, batch_info, args, all_numseq)
                            model_engine.module.enable_adapter_layers()

                        combined_t, num_chosen = _combine_chosen_rejected_batch(chosen_t, rejected_t)
                        per_seq_logps = model_engine(
                            **combined_t, **batch_info,
                            num_seq=0,
                            return_per_seq_logps=True,
                            chunk_size=args.chunk_size,
                            use_fast_norm=args.use_fast_norm,
                            use_fast_rope=args.use_fast_rope,
                        ).logits
                        chosen_logp = per_seq_logps[:num_chosen]
                        rejected_logp = per_seq_logps[num_chosen:]
                        eval_loss = dpo_loss(chosen_logp, rejected_logp, chosen_ref, rejected_ref, args.dpo_beta)

                        eval_total_loss.add_(eval_loss)
                        eval_total_steps += 1

                eval_total_loss.div_(eval_total_steps)
                dist.reduce(eval_total_loss, 0)
                eval_epoch += 1

                if RANK == 0:
                    mlflow.log_metrics(metrics={"eval/loss": eval_total_loss.item()}, step=step)
                model_engine.train()

            if ((step == train_total_steps) or (epoch + 1 == args.epochs) or
                (args.save_strategy == "epoch" and args.save_every and ((epoch + 1) % args.save_every == 0))):
                dist.barrier()
                state_dict = _save_state_dict(model_engine, args)
                if RANK == 0 and state_dict is not None:
                    save_path = os.path.join(args.save_path, f"ep_{epoch + 1}")
                    model_engine.module.save_pretrained(save_path, state_dict=state_dict)
                    save_tokenizer(args, save_path)
                    save_openchat_metadata(args, epoch + 1, step, save_path)

    if RANK == 0:
        progress_bar.close()
        save_path = args.save_path
        model_engine.module.save_pretrained(save_path, state_dict=state_dict)
        save_tokenizer(args, save_path)
        save_openchat_metadata(args, epoch + 1, step, save_path)
        mlflow.end_run()


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    with open(args_dict["deepspeed_config"]) as f:
        deepspeed_config: dict = json.load(f)
    args_dict["ds_zero_op"] = deepspeed_config.get("zero_optimization", {}).get("stage", 2)
    if deepspeed_config.get("zero_optimization", {}).get("offload_optimizer", False) or \
       deepspeed_config.get("zero_optimization", {}).get("offload_param", False):
        args_dict["ds_offload"] = True
        args_dict["use_zero_one_opt"] = False
        args_dict["use_qlora"] = False
    args = TrainingArguments(**args_dict)
    train(args)
