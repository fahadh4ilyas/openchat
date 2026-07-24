"""
Generate ORPO training data from paired chosen/rejected conversation JSONL.

ORPO does not require a reference model, so this is identical to the DPO
pipeline except reference log-prob computation is always skipped
(saves GPU memory and time).

Usage: python -m ochat.data.generate_orpo_dataset \
    --model-type MODEL_TYPE --model-path BASE_REPO \
    --in-files data_orpo.jsonl --out-prefix PRETOKENIZED_ORPO_DATA_OUTPUT_PATH

Input JSONL format (each line):
{
  "chosen":  {"items": [...], "system": "..."},
  "rejected": {"items": [...], "system": "..."}
}

Output: Parquet files with tokenized chosen_*/rejected_* fields.
The chosen_ref_logp and rejected_ref_logp fields contain NaN sentinels
(the ORPO trainer ignores them).
"""

import sys
from ochat.data.generate_dpo_dataset import DataArguments, generate_dataset


if __name__ == "__main__":
    # ORPO always skips ref log-probs — remove --ref-logps if present
    args_list = [a for a in sys.argv[1:] if a != "--ref-logps"]

    # Reuse DPO data generation pipeline (same JSONL format, same tokenization)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--in-files", type=str, nargs="+", required=True)
    parser.add_argument("--out-prefix", type=str, required=True)
    parser.add_argument("--per-sequence-loss", action="store_true")
    parser.add_argument("--force-eos-token", action="store_true")
    parser.add_argument("--eos-final", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--ignore-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.0)
    parser.add_argument("--data-length-multiple-of", type=int, default=1)
    parser.add_argument("--ignore-last-token", action="store_true")
    parser.add_argument("--separate-think", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=10)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--split-files", action="store_true")

    args = parser.parse_args(args_list)
    args = DataArguments(**vars(args), ref_logps=False)

    generate_dataset(args)
