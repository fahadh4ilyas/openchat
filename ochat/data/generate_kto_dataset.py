"""
Generate KTO training data from conversation JSONL.

KTO uses unpaired data with a `label` field (true = desirable,
false = undesirable). This is a thin wrapper around generate_dataset.py
with --kto preset.

Usage: python -m ochat.data.generate_kto_dataset \
    --model-type MODEL_TYPE --model-path BASE_REPO \
    --in-files data_kto.jsonl --out-prefix PRETOKENIZED_KTO_DATA_OUTPUT_PATH

To precompute reference log-probs during preprocessing:
    python -m ochat.data.generate_kto_dataset ... --ref-logps

Input JSONL format (each line is a Conversation with a `label` field):
    {"items": [...], "label": true}
    {"items": [...], "label": false}

Output: Parquet files with standard SFT tokenization fields plus
`label` (bool) and `ref_logp` (float32, NaN sentinel if --ref-logps
was not used).
"""

from ochat.data.generate_dataset import DataArguments, generate_dataset


if __name__ == "__main__":
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
    parser.add_argument("--pretokenized-in-files", action="store_true")
    parser.add_argument("--pretraining-data", action="store_true")
    parser.add_argument("--ignore-last-token", action="store_true")
    parser.add_argument("--separate-think", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=10)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--split-files", action="store_true")
    parser.add_argument("--ref-logps", action="store_true",
                        help="Compute reference log-probs during preprocessing (requires GPU)")

    args = parser.parse_args()
    args = DataArguments(**vars(args), kto=True)

    generate_dataset(args)
