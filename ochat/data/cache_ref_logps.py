"""
Precompute reference log-probs for an already-tokenized dataset (DPO or KTO).

Reads pretokenized Arrow/Parquet files and runs the base model over each
example to compute per-example reference log-probabilities. Saves cache
files that training scripts can load to avoid online computation.

Auto-detects dataset format:
    - Has column "chosen_ref_logp" → DPO (chosen + rejected pair)
    - Has column "label" → KTO (unpaired desirable/undesirable)

Usage:
    python -m ochat.data.cache_ref_logps \
        --data-prefix PRETOKENIZED_DATA_PATH \
        --model-path BASE_REPO

Output:
    DPO: {data_prefix}.{split}.ref_logps_cache.npz
    KTO: {data_prefix}.{split}.kto_ref_logps_cache.npz

Each cache file contains a checksum so training scripts can validate
they match the current dataset.
"""

import argparse
import os

import numpy as np
import torch

from ochat.config import MODEL_CONFIG_MAP
from ochat.training_utils.numpy_dataset import NumpyDataset
from ochat.training_utils._common import batch_to_tensor, _compute_dataset_checksum


def _detect_format(dataset: NumpyDataset) -> str:
    """Auto-detect whether the dataset is DPO or KTO format."""
    try:
        _ = dataset["chosen_ref_logp"]
        return "dpo"
    except KeyError:
        pass
    try:
        _ = dataset["label"]
        return "kto"
    except KeyError:
        pass
    raise ValueError("Cannot detect dataset format: no 'chosen_ref_logp' (DPO) or 'label' (KTO) column found.")


def _compute_dpo(model, dataset: NumpyDataset, device: torch.device,
                 processor=None, dataset_path=None):
    """Compute DPO reference log-probs for chosen and rejected sides."""
    num_examples = len(dataset)
    chosen_ref = np.empty(num_examples, dtype=np.float32)
    rejected_ref = np.empty(num_examples, dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for i in range(num_examples):
            example = dataset[[i]]

            for side, out in (("chosen", chosen_ref), ("rejected", rejected_ref)):
                tensor, info = batch_to_tensor(example, dataset_path, processor, prefix=f"{side}_")
                tensor = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                          for k, v in tensor.items()}

                loss = model(**tensor, **info, num_seq=1).loss
                if isinstance(loss, tuple):
                    loss, _ = loss
                out[i] = -loss.item()

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{num_examples} pairs")
                torch.cuda.empty_cache()

    return chosen_ref, rejected_ref


def _compute_kto(model, dataset: NumpyDataset, device: torch.device,
                 processor=None, dataset_path=None):
    """Compute KTO reference log-probs for single-side examples."""
    num_examples = len(dataset)
    ref_logp = np.empty(num_examples, dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for i in range(num_examples):
            example = dataset[[i]]
            tensor, info = batch_to_tensor(example, dataset_path, processor)
            tensor = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in tensor.items()}

            output = model(**tensor, **info, num_seq=0,
                           return_per_seq_logps=True,
                           use_fast_norm=False,
                           use_fast_rope=False)
            ref_logp[i] = output.logits.sum().item()

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{num_examples} examples")
                torch.cuda.empty_cache()

    return ref_logp


def main():
    parser = argparse.ArgumentParser(
        description="Precompute reference log-probs for a pretokenized DPO or KTO dataset."
    )
    parser.add_argument("--data-prefix", "--data_prefix", type=str, required=True,
                        help="Path prefix to pretokenized .parquet files")
    parser.add_argument("--model-path", "--model_path", type=str, required=True,
                        help="HuggingFace repo or local path to the base model")
    args = parser.parse_args()

    # Load train split (required)
    train_file = f"{args.data_prefix}.train"
    if not (os.path.isfile(train_file + ".parquet") or os.path.isfile(train_file + ".pickle")):
        raise FileNotFoundError(f"Training data not found at {train_file}")

    print(f"Loading train data from {train_file}...")
    train_dataset = NumpyDataset(train_file)
    fmt = _detect_format(train_dataset)
    print(f"  Detected format: {fmt}")
    model_type = train_dataset.metadata["model_type"]

    # Load eval split (optional)
    eval_dataset = None
    eval_file = f"{args.data_prefix}.eval"
    if os.path.isfile(eval_file + ".parquet") or os.path.isfile(eval_file + ".pickle"):
        print(f"Loading eval data from {eval_file}...")
        eval_dataset = NumpyDataset(eval_file)

    # Load model
    print(f"Loading model {model_type} from {args.model_path}...")
    model_config = MODEL_CONFIG_MAP[model_type]
    model = model_config.model_create_for_training(
        args.model_path, dtype=torch.bfloat16,
    )
    model = model.to("cuda")
    model.config.use_cache = False

    # Load processor for multimodal support (images/videos)
    processor = None
    dataset_path = os.path.dirname(args.data_prefix) or "."
    if model_config.model_has_processor:
        tokenizer = model_config.model_tokenizer_create(args.model_path)
        processor = tokenizer
        print(f"  Loaded processor for multimodal data")

    # Process each split
    splits = [("train", train_dataset)]
    if eval_dataset is not None:
        splits.append(("eval", eval_dataset))

    for split_name, dataset in splits:
        print(f"\nComputing reference log-probs for {split_name} split ({len(dataset)} examples)...")

        if fmt == "dpo":
            chosen, rejected = _compute_dpo(model, dataset, model.device,
                                            processor=processor, dataset_path=dataset_path)
            checksum = _compute_dataset_checksum(dataset, ["chosen_nz_input_ids", "rejected_nz_input_ids"])
            cache_path = f"{args.data_prefix}.{split_name}.ref_logps_cache.npz"
            np.savez(cache_path, checksum=checksum, chosen_ref_logp=chosen, rejected_ref_logp=rejected)
            print(f"  Saved {cache_path} (checksum {checksum}, {len(chosen)} pairs)")

        else:  # kto
            ref_logp = _compute_kto(model, dataset, model.device,
                                    processor=processor, dataset_path=dataset_path)
            checksum = _compute_dataset_checksum(dataset, ["nz_input_ids"])
            cache_path = f"{args.data_prefix}.{split_name}.kto_ref_logps_cache.npz"
            np.savez(cache_path, checksum=checksum, ref_logp=ref_logp)
            print(f"  Saved {cache_path} (checksum {checksum}, {len(ref_logp)} examples)")

    print("\nDone.")


if __name__ == "__main__":
    main()
