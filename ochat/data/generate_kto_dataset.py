"""
Generate KTO training data from conversation JSONL.

KTO uses unpaired data with a `label` field (true = desirable,
false = undesirable). This is a thin wrapper around generate_dataset.py
with --kto preset.

Usage: python -m ochat.data.generate_kto_dataset \
    --model-type MODEL_TYPE --model-path BASE_REPO \
    --in-files data_kto.jsonl --out-prefix PRETOKENIZED_KTO_DATA_OUTPUT_PATH

Reference log-probs are computed separately via cache_ref_logps.py,
not during preprocessing.

Input JSONL format (each line is a Conversation with a `label` field):
    {"items": [...], "label": true}
    {"items": [...], "label": false}

Output: Parquet files with standard SFT tokenization fields plus
`label` (bool) and `ref_logp` (float32, NaN sentinel — filled later
by cache_ref_logps.py).
"""

from ochat.data.generate_dataset import DataArguments, generate_dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", "--model_type", type=str, required=True)
    parser.add_argument("--model-path", "--model_path", type=str, required=True)
    parser.add_argument("--in-files", "--in_files", type=str, nargs="+", required=True)
    parser.add_argument("--out-prefix", "--out_prefix", type=str, required=True)
    parser.add_argument("--per-sequence-loss", "--per_sequence_loss", action="store_true")
    parser.add_argument("--force-eos-token", "--force_eos_token", action="store_true")
    parser.add_argument("--eos-final", "--eos_final", action="store_true")
    parser.add_argument("--max-seq-length", "--max_seq_length", type=int, default=None)
    parser.add_argument("--ignore-index", "--ignore_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", "--eval_ratio", type=float, default=0.0)
    parser.add_argument("--data-length-multiple-of", "--data_length_multiple_of", type=int, default=1)
    parser.add_argument("--pretokenized-in-files", "--pretokenized_in_files", action="store_true")
    parser.add_argument("--pretraining-data", "--pretraining_data", action="store_true")
    parser.add_argument("--ignore-last-token", "--ignore_last_token", action="store_true")
    parser.add_argument("--separate-think", "--separate_think", action="store_true")
    parser.add_argument("--max-workers", "--max_workers", type=int, default=None)
    parser.add_argument("--max-jobs", "--max_jobs", type=int, default=10)
    parser.add_argument("--num-splits", "--num_splits", type=int, default=10)
    parser.add_argument("--split-files", "--split_files", action="store_true")

    args = parser.parse_args()
    args = DataArguments(**vars(args), kto=True)

    generate_dataset(args)


if __name__ == "__main__":
    main()
