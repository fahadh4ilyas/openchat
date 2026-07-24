"""Shared training loop infrastructure for SFT, DPO, and ORPO.

Consolidates duplicated code from 9 training scripts (3 modes × 3 variants).
Each mode's train.py imports these and provides its mode-specific forward/loss.
"""

import os
import json
import hashlib

import torch
import torch.distributed as dist
import numpy as np

import mlflow


# -- Model creation (shared by all distributed training scripts) ---------------


def create_model_and_engine(args, base_lr: float) -> tuple:
    """Create model with optional LoRA/QLoRA and DeepSpeed engine.

    Shared by SFT, DPO, and ORPO distributed training.

    Args:
        args: TrainingArguments with model_path, model_type, lora/q-lora fields,
              deepspeed_config, optimizer fields.
        base_lr: Base learning rate (used to report and for lr auto-estimation).

    Returns:
        (model_engine, optimizer) tuple.
    """
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    from ochat.config import MODEL_CONFIG_MAP
    from ochat.training_utils._common import get_latest_checkpoint

    try:
        import deepspeed
    except ImportError:
        raise ImportError("Please install deepspeed to train models.")

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

    is_lora = args.use_lora or args.use_qlora

    if is_lora:
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
            "params": {"lr": args.lr, "weight_decay": args.weight_decay,
                       "betas": [args.beta1, args.beta2], "eps": args.eps},
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


# -- Checkpoint helpers --------------------------------------------------------


def save_state_dict(model_engine) -> dict | None:
    """Save model state dict, handling ZeRO stage 2 vs 3.

    Args:
        model_engine: DeepSpeed engine.

    Returns:
        State dict on rank 0 (or all ranks for ZeRO-3), None on other ranks.
    """
    import deepspeed
    if model_engine.zero_optimization_stage() == 3:
        return model_engine._zero3_consolidated_16bit_state_dict()
    elif dist.get_rank() == 0:
        return deepspeed.checkpoint.utils.clone_tensors_for_torch_save(
            model_engine.module.state_dict()
        )
    return None


def save_checkpoint(model_engine, args, save_path: str, epoch: float, step: int,
                    save_tokenizer_too: bool = False):
    """Save a training checkpoint (model + optional tokenizer + metadata).

    Args:
        model_engine: DeepSpeed engine.
        args: Training arguments.
        save_path: Directory to save to.
        epoch: Current epoch (float for fractional epochs).
        step: Current step.
        save_tokenizer_too: If True, also save the tokenizer.
    """
    from ochat.training_utils._common import save_openchat_metadata, save_tokenizer
    state_dict = save_state_dict(model_engine)
    if dist.get_rank() == 0 and state_dict is not None:
        model_engine.module.save_pretrained(save_path, state_dict=state_dict)
        if save_tokenizer_too:
            save_tokenizer(args, save_path)
        save_openchat_metadata(args, epoch, step, save_path)


# -- MLflow -------------------------------------------------------------------


def setup_mlflow(args, train_total_steps: int, RANK: int):
    """Initialize MLflow tracking on rank 0.

    Args:
        args: Training arguments (tracking_uri, experiment_name, run_name, mlflow creds).
        train_total_steps: Total training steps (logged as param).
        RANK: Current process rank.
    """
    if RANK != 0:
        return
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


# -- Eval helpers -------------------------------------------------------------


def run_eval(
    model_engine, eval_loader, args, eval_epoch: int,
    eval_step_fn, step: int | None = None
) -> tuple:
    """Run one evaluation pass.

    Args:
        model_engine: DeepSpeed engine.
        eval_loader: Validation dataloader.
        args: Training arguments.
        eval_epoch: Current eval epoch counter.
        eval_step_fn: Callable(model_engine, batch, args) -> loss_tensor.
            Must handle device transfer internally.
        step: Current training step (for MLflow logging). If None, no logging.

    Returns:
        (avg_loss_tensor, next_eval_epoch) tuple.
    """
    eval_total_loss = torch.zeros((), dtype=torch.float32, device=args.device)
    eval_total_steps = 0

    model_engine.eval()
    eval_loader.set_epoch(eval_epoch)
    with torch.inference_mode():
        for batch in eval_loader:
            eval_loss = eval_step_fn(model_engine, batch, args)
            eval_total_loss.add_(eval_loss)
            eval_total_steps += 1

    eval_total_loss.div_(eval_total_steps)
    dist.reduce(eval_total_loss, 0)
    model_engine.train()

    if dist.get_rank() == 0 and step is not None:
        mlflow.log_metrics(
            metrics={"eval/loss": eval_total_loss.item()}, step=step
        )

    return eval_total_loss, eval_epoch + 1


# -- Mode-specific arg post-processing ---------------------------------------


def _parse_ds_config(args_dict: dict) -> dict:
    """Parse DeepSpeed config from args, setting ds_zero_op and ds_offload.

    Mutates args_dict in place and returns the modified dict.
    """
    with open(args_dict["deepspeed_config"]) as f:
        deepspeed_config: dict = json.load(f)
    args_dict["ds_zero_op"] = deepspeed_config.get("zero_optimization", {}).get("stage", 2)
    if (deepspeed_config.get("zero_optimization", {}).get("offload_optimizer", False)
            or deepspeed_config.get("zero_optimization", {}).get("offload_param", False)):
        args_dict["ds_offload"] = True
        args_dict["use_zero_one_opt"] = False
        args_dict["use_qlora"] = False
    return args_dict


# -- DPO reference log-prob cache --------------------------------------------

_CACHE_SAMPLE_TOKENS = 256  # first N tokens hashed per head/tail example


def _dataset_checksum(dataset) -> str:
    """Compact fingerprint of dataset contents for cache validation.

    Hashes:  dataset length + first/last example's input IDs + total token count.
    Only ranks first/last samples and a few hundred tokens, so is fast even
    on large datasets.
    """
    h = hashlib.sha256()
    h.update(str(len(dataset)).encode())

    for idx in (0, len(dataset) - 1):
        for prefix in ("chosen_", "rejected_"):
            arr = dataset[f"{prefix}nz_input_ids"][idx]
            h.update(arr[: _CACHE_SAMPLE_TOKENS].tobytes())

    h.update(str(int(dataset["total_length"].sum())).encode())
    return h.hexdigest()[:16]


def ensure_dpo_ref_logps_cached(model_engine, dataset, args, split_name: str):
    """Precompute and cache DPO reference log-probs at training start.

    If reference log-probs were not precomputed during data preprocessing,
    this runs a single pass over the dataset using the frozen base model
    (LoRA adapters disabled) and caches the results to disk.  Subsequent
    training runs load from cache, avoiding per-batch online computation.

    A dataset checksum is embedded in the cache file — if the dataset
    changes, the cache is automatically invalidated and recomputed.

    Works for both distributed (DeepSpeed engine) and single-GPU (plain model).

    Args:
        model_engine: DeepSpeed engine or plain PeftModel.
        dataset: NumpyDataset for the split.
        args: Training arguments.
        split_name: "train" or "eval".

    Returns:
        (chosen_ref_logp, rejected_ref_logp) numpy arrays of shape (N,).
    """
    from ochat.training_utils._common import batch_to_tensor

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    model = model_engine.module if hasattr(model_engine, "module") else model_engine
    checksum = _dataset_checksum(dataset)
    cache_path = f"{args.data_prefix}.{split_name}.ref_logps_cache.npz"

    # Try loading cached result (rank 0 loads from disk, broadcasts to all)
    cache_hit = False
    if rank == 0 and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if str(cached.get("checksum", "")) == checksum:
            chosen_ref = cached["chosen_ref_logp"]
            rejected_ref = cached["rejected_ref_logp"]
            cache_hit = True
            print(f"Loaded cached reference log-probs from {cache_path} "
                  f"(checksum {checksum})")
        else:
            print(f"Cache {cache_path} is stale (checksum mismatch "
                  f"{cached.get('checksum', 'none')} vs {checksum}), "
                  f"recomputing...")

    if is_distributed:
        cache_hit_t = torch.tensor([cache_hit], device=args.device)
        dist.broadcast(cache_hit_t, src=0)
        cache_hit = bool(cache_hit_t.item())

        if cache_hit:
            # Broadcast arrays from rank 0
            if rank == 0:
                size_t = torch.tensor([len(chosen_ref)], device=args.device)
            else:
                size_t = torch.empty(1, dtype=torch.long, device=args.device)
            dist.broadcast(size_t, src=0)
            n = size_t.item()

            if rank != 0:
                chosen_ref = np.empty(n, dtype=np.float32)
                rejected_ref = np.empty(n, dtype=np.float32)
            chosen_t = torch.from_numpy(chosen_ref).to(args.device)
            rejected_t = torch.from_numpy(rejected_ref).to(args.device)
            dist.broadcast(chosen_t, src=0)
            dist.broadcast(rejected_t, src=0)
            if rank != 0:
                chosen_ref = chosen_t.cpu().numpy()
                rejected_ref = rejected_t.cpu().numpy()
            return chosen_ref, rejected_ref

    if rank == 0:
        print(f"Precomputing reference log-probs for {split_name} split "
              f"(checksum {checksum}, caching to {cache_path})...")

    num_examples = len(dataset)
    chosen_ref = np.empty(num_examples, dtype=np.float32)
    rejected_ref = np.empty(num_examples, dtype=np.float32)

    model.disable_adapter_layers()

    with torch.no_grad():
        for i in range(num_examples):
            example = dataset[[i]]  # list index preserves batch structure

            chosen_t, chosen_info = batch_to_tensor(example, prefix="chosen_")
            chosen_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                        for k, v in chosen_t.items()}
            chosen_logp = model_engine(
                **chosen_t, **chosen_info,
                num_seq=0, return_per_seq_logps=True,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).logits.sum().item()
            chosen_ref[i] = chosen_logp

            rejected_t, rejected_info = batch_to_tensor(example, prefix="rejected_")
            rejected_t = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                          for k, v in rejected_t.items()}
            rejected_logp = model_engine(
                **rejected_t, **rejected_info,
                num_seq=0, return_per_seq_logps=True,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).logits.sum().item()
            rejected_ref[i] = rejected_logp

    model.enable_adapter_layers()

    # Save to disk on rank 0 for future runs
    if rank == 0:
        np.savez(cache_path,
                 checksum=checksum,
                 chosen_ref_logp=chosen_ref,
                 rejected_ref_logp=rejected_ref)

    # Broadcast to all ranks (works without shared filesystem)
    if is_distributed:
        chosen_t = torch.from_numpy(chosen_ref).to(args.device)
        rejected_t = torch.from_numpy(rejected_ref).to(args.device)
        dist.broadcast(chosen_t, src=0)
        dist.broadcast(rejected_t, src=0)
        if rank != 0:
            chosen_ref = chosen_t.cpu().numpy()
            rejected_ref = rejected_t.cpu().numpy()

    return chosen_ref, rejected_ref


def ensure_kto_ref_logps_cached(model_engine, dataset, args, split_name: str):
    """Precompute and cache KTO reference log-probs at training start.

    Same pattern as ensure_dpo_ref_logps_cached but for unpaired KTO data.
    Each example has a single ref_logp (no chosen/rejected prefix).

    Args:
        model_engine: DeepSpeed engine or plain PeftModel.
        dataset: NumpyDataset for the split.
        args: Training arguments.
        split_name: "train" or "eval".

    Returns:
        ref_logp numpy array of shape (N,).
    """
    from ochat.training_utils._common import batch_to_tensor

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    model = model_engine.module if hasattr(model_engine, "module") else model_engine
    checksum = _dataset_checksum_kto(dataset)
    cache_path = f"{args.data_prefix}.{split_name}.kto_ref_logps_cache.npz"

    cache_hit = False
    if rank == 0 and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if str(cached.get("checksum", "")) == checksum:
            ref_logp = cached["ref_logp"]
            cache_hit = True
            print(f"Loaded cached KTO reference log-probs from {cache_path} "
                  f"(checksum {checksum})")
        else:
            print(f"Cache {cache_path} is stale (checksum mismatch "
                  f"{cached.get('checksum', 'none')} vs {checksum}), "
                  f"recomputing...")

    if is_distributed:
        cache_hit_t = torch.tensor([cache_hit], device=args.device)
        dist.broadcast(cache_hit_t, src=0)
        cache_hit = bool(cache_hit_t.item())

        if cache_hit:
            if rank == 0:
                size_t = torch.tensor([len(ref_logp)], device=args.device)
            else:
                size_t = torch.empty(1, dtype=torch.long, device=args.device)
            dist.broadcast(size_t, src=0)
            n = size_t.item()

            if rank != 0:
                ref_logp = np.empty(n, dtype=np.float32)
            ref_t = torch.from_numpy(ref_logp).to(args.device)
            dist.broadcast(ref_t, src=0)
            if rank != 0:
                ref_logp = ref_t.cpu().numpy()
            return ref_logp

    if rank == 0:
        print(f"Precomputing KTO reference log-probs for {split_name} split "
              f"(checksum {checksum}, caching to {cache_path})...")

    num_examples = len(dataset)
    ref_logp = np.empty(num_examples, dtype=np.float32)

    model.disable_adapter_layers()

    with torch.no_grad():
        for i in range(num_examples):
            example = dataset[[i]]
            tensor, info = batch_to_tensor(example)
            tensor = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                      for k, v in tensor.items()}

            loss = model_engine(
                **tensor, **info,
                num_seq=0, return_per_seq_logps=True,
                use_fast_norm=args.use_fast_norm,
                use_fast_rope=args.use_fast_rope,
            ).logits
            ref_logp[i] = loss.sum().item()

    model.enable_adapter_layers()

    if rank == 0:
        np.savez(cache_path, checksum=checksum, ref_logp=ref_logp)

    if is_distributed:
        ref_t = torch.from_numpy(ref_logp).to(args.device)
        dist.broadcast(ref_t, src=0)
        if rank != 0:
            ref_logp = ref_t.cpu().numpy()

    return ref_logp


def _dataset_checksum_kto(dataset) -> str:
    """Compact fingerprint of KTO dataset contents for cache validation."""
    import hashlib
    h = hashlib.sha256()
    h.update(str(len(dataset)).encode())

    for idx in (0, len(dataset) - 1):
        arr = dataset["nz_input_ids"][idx]
        h.update(arr[:_CACHE_SAMPLE_TOKENS].tobytes())

    h.update(str(int(dataset["total_length"].sum())).encode())
    return h.hexdigest()[:16]
