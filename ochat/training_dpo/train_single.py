"""Single-GPU DPO training entry point.

Supports full fine-tuning when ref log-probs are precomputed; LoRA/QLoRA
required when computing them at training start (the frozen base model
serves as the reference model).

base_lr=3e-4 (full FT), 1e-2 (LoRA).
"""

import argparse
import os
from functools import partial

import torch

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
from ochat.training_utils._common import (
    combine_chosen_rejected_batch,
    _check_ref_logps_ready,
    per_seq_response_tokens,
    mlflow_stopper_wrapper,
    get_latest_checkpoint,
    create_dataset,
    calculate_auto_lr,
    create_lr_scheduler,
    save_openchat_metadata,
    clean_checkpoint,
    save_tokenizer,
    load_tokenizer,
)
from ochat.training_utils.base_train import ensure_dpo_ref_logps_cached
from ochat.training_dpo.utils import (
    dpo_batch_collate,
    dpo_loss_router,
)
from ochat.training_utils.multipack_dataloader_single import MultipackDataloader
from ochat.training_utils.numpy_dataset import NumpyDataset

from transformers import BitsAndBytesConfig


class TrainingArguments(BaseTrainingArguments, LoraTrainingArgsMixin):
    """DPO training arguments."""
    dpo_beta: float = 0.1
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0
    discopop_tau: float = 0.05
    cpo_alpha: float = 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    add_base_args(parser, base_lr=3e-4)
    add_lora_args(parser)
    parser.add_argument("--dpo-beta", "--dpo_beta", type=float, default=0.1, help="DPO temperature parameter")
    parser.add_argument("--loss-type", "--loss_type", type=str, default="sigmoid",
                        help="DPO loss variant: sigmoid, hinge, ipo, exo_pair, nca_pair, robust, "
                             "bco_pair, sppo_hard, aot, aot_unpaired, apo_zero, apo_down, discopop, "
                             "sft, sigmoid_norm")
    parser.add_argument("--label-smoothing", "--label_smoothing", type=float, default=0.0,
                        help="Label smoothing ε (for exo_pair, robust, aot, aot_unpaired)")
    parser.add_argument("--discopop-tau", "--discopop_tau", type=float, default=0.05,
                        help="DiscoPOP temperature τ")
    parser.add_argument("--cpo-alpha", "--cpo_alpha", type=float, default=0.0,
                        help="CPO SFT weight (0 = pure DPO, >0 = CPO. Default: 0)")
    return parser.parse_args()


def create_distributed_dataloader(args, data: NumpyDataset):
    collate_fn = dpo_batch_collate
    if args.has_processor:
        tokenizer = load_tokenizer(args)
        collate_fn = partial(dpo_batch_collate, dataset_path=os.path.dirname(args.data_prefix), processor=tokenizer)
    return MultipackDataloader(
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
        model_path, low_cpu_mem_usage=True, quantization_config=quantization_config,
    )

    model.config.use_cache = False
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    is_lora = args.use_lora or args.use_qlora
    if is_lora:
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

    model = model.to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), eps=args.eps, fused=True,
    )

    args.device = model.device
    return model, optimizer


def _eval_loop(model, eval_loader, args, eval_epoch):
    model.eval()
    eval_total_loss = torch.zeros((), dtype=torch.float32, device=args.device)
    eval_total_steps = 0

    need_tokens = args.loss_type in ("ipo", "sigmoid_norm")
    use_cpo = getattr(args, "cpo_alpha", 0.0) > 0

    eval_loader.set_epoch(eval_epoch)
    with torch.inference_mode():
        for (chosen_t, rejected_t, chosen_ref, rejected_ref, batch_info), num_seq in eval_loader:
            chosen_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in chosen_t.items()}
            rejected_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in rejected_t.items()}
            chosen_ref = chosen_ref.to(args.device)
            rejected_ref = rejected_ref.to(args.device)

            combined_t, num_chosen = combine_chosen_rejected_batch(chosen_t, rejected_t)
            outputs = model(
                **combined_t, **batch_info,
                num_seq=0, return_per_seq_logps=True,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            )
            per_seq_logps = outputs.logits
            chosen_logp = per_seq_logps[:num_chosen]
            rejected_logp = per_seq_logps[num_chosen:]

            kwargs = {}
            if need_tokens:
                resp_tokens = per_seq_response_tokens(combined_t)
                kwargs["chosen_tokens"] = resp_tokens[:num_chosen]
                kwargs["rejected_tokens"] = resp_tokens[num_chosen:]

            if use_cpo:
                dpo_term = dpo_loss_router(
                    chosen_logp, rejected_logp, chosen_ref, rejected_ref,
                    args.dpo_beta, loss_type="sigmoid", **kwargs,
                )
                eval_loss = dpo_term + args.cpo_alpha * (-chosen_logp.mean())
            else:
                eval_loss = dpo_loss_router(
                    chosen_logp, rejected_logp, chosen_ref, rejected_ref,
                    args.dpo_beta, loss_type=args.loss_type,
                    label_smoothing=args.label_smoothing,
                    discopop_tau=args.discopop_tau,
                    **kwargs,
                )
            # Add MoE aux_loss for consistent train/eval comparison
            if outputs.loss is not None:
                loss_struct, _ = outputs.loss
                if isinstance(loss_struct, tuple):
                    eval_loss = eval_loss + loss_struct[1]
            eval_total_loss.add_(eval_loss)
            eval_total_steps += 1

    return eval_total_loss / eval_total_steps, eval_epoch + 1


@mlflow_stopper_wrapper(is_distributed=False)
def train(args):
    use_cpo = getattr(args, "cpo_alpha", 0.0) > 0
    need_tokens = args.loss_type in ("ipo", "sigmoid_norm")

    train_dataset = create_dataset(args, "train")
    eval_dataset = create_dataset(args, "eval")

    if train_dataset is None:
        raise RuntimeError("Training data not found.")

    args.model_type = train_dataset.metadata["model_type"]
    args.has_processor = MODEL_CONFIG_MAP[args.model_type].model_has_processor

    is_lora = args.use_lora or args.use_qlora
    args.base_lr = 1e-2 if is_lora else 3e-4

    ref_logps_precomputed = _check_ref_logps_ready(
        train_dataset, eval_dataset, args,
        key="chosen_ref_logp",
        cache_suffix="ref_logps_cache.npz",
        checksum_keys=["chosen_nz_input_ids", "rejected_nz_input_ids"],
    )

    if not ref_logps_precomputed and not is_lora:
        raise RuntimeError(
            "DPO requires reference log-probs. They were not precomputed during preprocessing "
            "(use --ref-logps with generate_dpo_dataset.py). "
            "Without precomputed ref log-probs, training must compute them from the frozen base model, "
            "which requires LoRA/QLoRA (--use-lora or --use-qlora)."
        )

    train_loader = create_distributed_dataloader(args, train_dataset)
    if args.max_steps > 0:
        args.epochs = -(-args.max_steps // train_loader.num_batches())
        train_total_steps = args.max_steps
    else:
        train_total_steps = args.epochs * train_loader.num_batches()

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = create_distributed_dataloader(args, eval_dataset)

    args.lr = calculate_auto_lr(args.base_lr, args.lr, args.batch_max_len, train_dataset,
                                MODEL_CONFIG_MAP[args.model_type].base_lr_scale)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    if args.mlflow_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = args.mlflow_username
    if args.mlflow_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = args.mlflow_password
    mlflow.set_experiment(args.experiment_name)
    mlflow.start_run(run_name=args.run_name)
    metadata = vars(args).copy()
    metadata.pop("device", None)
    metadata["steps"] = train_total_steps
    mlflow.log_params(metadata)

    model, optimizer = create_model(args)

    # Load or compute reference log-probs (cache-loaded if precomputed, computed online if not)
    chosen_cache, rejected_cache = ensure_dpo_ref_logps_cached(
        model, train_dataset, args, "train"
    )
    train_dataset.dataset["chosen_ref_logp"] = chosen_cache
    train_dataset.dataset["rejected_ref_logp"] = rejected_cache
    if eval_dataset is not None:
        chosen_ecache, rejected_ecache = ensure_dpo_ref_logps_cached(
            model, eval_dataset, args, "eval"
        )
        eval_dataset.dataset["chosen_ref_logp"] = chosen_ecache
        eval_dataset.dataset["rejected_ref_logp"] = rejected_ecache

    lr_scheduler = create_lr_scheduler(args, train_total_steps)

    progress_bar = tqdm.tqdm(total=train_total_steps)

    step = 0
    latest_checkpoint = int((get_latest_checkpoint(args) or "_0").split("_")[-1])
    lr_this_step = None
    model.train()
    eval_epoch = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        train_loader.set_epoch(epoch)
        for (chosen_t, rejected_t, chosen_ref, rejected_ref, batch_info), num_seq in train_loader:
            step += 1
            if step > train_total_steps:
                break
            elif step <= latest_checkpoint:
                progress_bar.update()
                continue

            optimizer.zero_grad()

            # Move to device
            chosen_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in chosen_t.items()}
            rejected_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in rejected_t.items()}

            chosen_ref = chosen_ref.to(args.device)
            rejected_ref = rejected_ref.to(args.device)

            # Combine chosen + rejected → single forward + split per-seq log-probs
            combined_t, num_chosen = combine_chosen_rejected_batch(chosen_t, rejected_t)
            outputs = model(
                **combined_t, **batch_info,
                num_seq=0,
                return_per_seq_logps=True,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            )
            per_seq_logps = outputs.logits

            chosen_logp = per_seq_logps[:num_chosen]
            rejected_logp = per_seq_logps[num_chosen:]

            kwargs = {}
            if need_tokens:
                resp_tokens = per_seq_response_tokens(combined_t)
                kwargs["chosen_tokens"] = resp_tokens[:num_chosen]
                kwargs["rejected_tokens"] = resp_tokens[num_chosen:]

            if use_cpo:
                dpo_component = dpo_loss_router(
                    chosen_logp, rejected_logp, chosen_ref, rejected_ref,
                    args.dpo_beta, loss_type="sigmoid", **kwargs,
                )
                sft_component = -chosen_logp.mean()
                loss = dpo_component + args.cpo_alpha * sft_component
            else:
                loss = dpo_loss_router(
                    chosen_logp, rejected_logp, chosen_ref, rejected_ref,
                    args.dpo_beta, loss_type=args.loss_type,
                    label_smoothing=args.label_smoothing,
                    discopop_tau=args.discopop_tau,
                    **kwargs,
                )

            # Add MoE router load-balancing aux_loss (SFT includes this; DPO/ORPO/KTO must too)
            if outputs.loss is not None:
                loss_struct, _ = outputs.loss
                if isinstance(loss_struct, tuple):
                    loss = loss + loss_struct[1]

            loss.backward()

            lr_this_step = args.lr * lr_scheduler(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_this_step

            optimizer.step()

            del chosen_t, rejected_t
            if args.torch_empty_cache_steps is not None and step % args.torch_empty_cache_steps == 0:
                torch.cuda.empty_cache()

            mlflow.log_metrics(
                metrics={
                    "train/loss": loss.item(),
                    "train/lr": lr_this_step,
                    "train/epoch": args.epochs * step / train_total_steps,
                    **({"train/dpo_loss": dpo_component.item(),
                        "train/sft_loss": sft_component.item()} if use_cpo else {}),
                },
                step=step,
            )
            progress_bar.update()

            if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0):
                save_path = os.path.join(args.save_path, f"checkpoint_{step}")
                model.save_pretrained(save_path)
                save_openchat_metadata(args, epoch + 1, step, save_path)
                clean_checkpoint(args)

            if args.save_strategy == "step" and args.save_every and (step % args.save_every == 0):
                save_path = os.path.join(args.save_path, f"st_{step}")
                model.save_pretrained(save_path)
                save_tokenizer(args, save_path)
                save_openchat_metadata(args, args.epochs * step / train_total_steps, step, save_path)

        if step > latest_checkpoint:
            mlflow.log_metrics(metrics={"batch_efficiency": train_loader.efficiency()}, step=step)

            if eval_loader is not None and (
                (step == train_total_steps) or (epoch + 1 == args.epochs) or
                (args.eval_strategy == "epoch" and args.eval_every and ((epoch + 1) % args.eval_every == 0))
            ):
                model.eval()
                eval_loss, eval_epoch = _eval_loop(model, eval_loader, args, eval_epoch)
                mlflow.log_metrics(metrics={"eval/loss": eval_loss.item()}, step=step)
                model.train()

            if args.save_strategy == "epoch" and args.save_every and ((epoch + 1) % args.save_every == 0):
                save_path = os.path.join(args.save_path, f"ep_{epoch + 1}")
                model.save_pretrained(save_path)
                save_tokenizer(args, save_path)
                save_openchat_metadata(args, epoch + 1, step, save_path)

    progress_bar.close()

    save_path = args.save_path
    model.save_pretrained(save_path)
    save_tokenizer(args, save_path)
    save_openchat_metadata(args, epoch + 1, step, save_path)

    mlflow.end_run()


if __name__ == "__main__":
    args = parse_args()
    args = TrainingArguments(**vars(args))
    train(args)
