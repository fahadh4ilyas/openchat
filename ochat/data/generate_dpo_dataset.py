"""
Generate DPO training data from paired chosen/rejected conversation JSONL.

Usage: python -m ochat.data.generate_dpo_dataset \
    --model-type MODEL_TYPE --model-path BASE_REPO \
    --in-files data_dpo.jsonl --out-prefix PRETOKENIZED_DPO_DATA_OUTPUT_PATH

Input JSONL format (each line):
{
  "chosen":  {"items": [...], "system": "..."},
  "rejected": {"items": [...], "system": "..."}
}

Output: Parquet files with tokenized chosen_*/rejected_* fields plus pre-computed
chosen_ref_logp/rejected_ref_logp reference model log-prob sums.
"""

import gc
import concurrent.futures
from typing import List, Optional
import argparse
from datetime import datetime
import random

from pydantic import BaseModel, Field, field_validator as validator, ValidationInfo

import orjson
import pyarrow
from pyarrow import parquet

import torch
import numpy as np


class DataArguments(BaseModel):
    model_path: str = Field(...)
    model_type: str = Field(...)
    in_files: List[str] = Field(...)
    out_prefix: str = Field(...)
    per_sequence_loss: bool = Field(False)
    force_eos_token: bool = Field(False)
    eos_final: bool = Field(False)
    max_seq_length: Optional[int] = Field(None)
    ignore_index: int = Field(0)
    seed: int = Field(42)
    eval_ratio: float = Field(0.0)
    data_length_multiple_of: int = Field(1)
    ignore_last_token: bool = Field(False)
    separate_think: bool = Field(False)
    max_workers: Optional[int] = Field(None)
    max_jobs: int = Field(10)
    split_files: bool = Field(False)
    num_splits: int = Field(10)
    no_ref_logps: bool = Field(False)

    @validator("max_jobs")
    def check_max_jobs(cls, v, info: ValidationInfo):
        if info.data.get("max_workers") is not None and v > info.data["max_workers"]:
            raise ValueError("max_jobs cannot be greater than max_workers")
        return v


PAD_TOKEN_ID = 0


class DPOPair(BaseModel):
    chosen: dict
    rejected: dict


def job_print(job_id: int, *args, **kwargs):
    print(
        f'[{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}] [JOB ID: {job_id}]',
        *args,
        **kwargs,
    )


def _split(a: list, n: int):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def truncate_trailing_zero_weighted(tokens: list, weights: list):
    non_zero_index = len(weights) - 1
    while non_zero_index >= 0 and weights[non_zero_index] == 0:
        non_zero_index -= 1
    return tokens[: non_zero_index + 1], weights[: non_zero_index + 1]


def _build_single_conv(tokens: list, weights: list, images: list, videos: list, args: DataArguments):
    """Build a single conversation's padded token data dict."""
    tokens, weights = truncate_trailing_zero_weighted(tokens, weights)
    if not tokens:
        return None

    LABEL_PAD_TOKEN_ID = args.ignore_index if args.ignore_index != PAD_TOKEN_ID else PAD_TOKEN_ID
    labels = [(t if w != 0 else LABEL_PAD_TOKEN_ID) for t, w in zip(tokens, weights)]

    labels = labels[1:]
    weights = weights[1:]
    if args.ignore_last_token:
        length = len(tokens) - 1
        last_token = [tokens[-1]]
        tokens = tokens[:-1]
    else:
        length = len(tokens)
        last_token = [PAD_TOKEN_ID]
        labels = labels + [LABEL_PAD_TOKEN_ID]
        weights = weights + [0.0]

    addition = -(-length // args.data_length_multiple_of) * args.data_length_multiple_of - length
    if addition > 0:
        tokens.extend(last_token + [PAD_TOKEN_ID] * (addition - 1))
        weights.extend([0.0] * addition)
        labels.extend([LABEL_PAD_TOKEN_ID] * addition)
    length = len(tokens)

    return {
        "total_length": length,
        "seqlens": [length],
        "nz_input_ids": tokens,
        "nz_position_ids": list(range(length)),
        "nz_shifted_label_ids": labels,
        "nz_shifted_loss_weights": weights,
        "images": images,
        "videos": videos,
    }


def convert_conversation_batch(job_id: int, batch: list, args: DataArguments):
    """Tokenize a batch of DPO pairs (CPU-only, runs in parallel)."""
    from ochat.config import MODEL_CONFIG_MAP, Conversation

    model_config = MODEL_CONFIG_MAP[args.model_type]
    tokenizer = model_config.model_tokenizer_create(args.model_path)

    if model_config.model_has_processor:
        if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
            conv_template = model_config.conversation_template(tokenizer=tokenizer.tokenizer)
        else:
            conv_template = model_config.conversation_template(tokenizer=tokenizer)
        image_processor = None
        video_processor = None
        if hasattr(tokenizer, "image_processor") and tokenizer.image_processor is not None:
            image_processor = tokenizer.image_processor
        if hasattr(tokenizer, "video_processor") and tokenizer.video_processor is not None:
            video_processor = tokenizer.video_processor
    else:
        conv_template = model_config.conversation_template(tokenizer=tokenizer)

    # Parse JSON
    job_print(job_id, "Decoding JSON ...")
    pairs: List[DPOPair] = [DPOPair(**orjson.loads(line)) for line in batch]

    # Tokenize all chosen and rejected together
    all_convs = []
    for pair in pairs:
        all_convs.append(Conversation(**pair.chosen))
        all_convs.append(Conversation(**pair.rejected))

    # Collect images/videos
    has_image = any([bool(c.images) for c in all_convs])
    has_video = any([bool(c.videos) for c in all_convs])
    images_list = [c.images if c.images else [] for c in all_convs]
    videos_list = [c.videos if c.videos else [] for c in all_convs]
    if has_image and image_processor is None:
        job_print(job_id, "Warning: The tokenizer does not have an image processor but the data contains images. The images will be ignored.")
    if has_video and video_processor is None:
        job_print(job_id, "Warning: The tokenizer does not have a video processor but the data contains videos. The videos will be ignored.")

    job_print(job_id, "Tokenizing ...")
    tokens_list, weights_list = conv_template.tokenize_conversations(
        all_convs,
        inference=False,
        seq_level_weight=args.per_sequence_loss,
        force_eos_token=args.force_eos_token,
        eos_final=args.eos_final,
        separate_think=args.separate_think,
    )

    max_context = args.max_seq_length or model_config.model_max_context

    outputs = []
    for i in range(0, len(tokens_list), 2):
        chosen_tokens = tokens_list[i][:max_context]
        chosen_weights = weights_list[i][:max_context]
        rejected_tokens = tokens_list[i + 1][:max_context]
        rejected_weights = weights_list[i + 1][:max_context]
        chosen_images = images_list[i] if images_list else []
        rejected_images = images_list[i + 1] if images_list else []
        chosen_videos = videos_list[i] if videos_list else []
        rejected_videos = videos_list[i + 1] if videos_list else []

        chosen_data = _build_single_conv(chosen_tokens, chosen_weights, chosen_images, chosen_videos, args)
        rejected_data = _build_single_conv(rejected_tokens, rejected_weights, rejected_images, rejected_videos, args)

        if chosen_data is None or rejected_data is None:
            continue

        row = {}
        for k in chosen_data:
            row[f"chosen_{k}"] = chosen_data[k]
        for k in rejected_data:
            row[f"rejected_{k}"] = rejected_data[k]

        row["total_length"] = chosen_data["total_length"] + rejected_data["total_length"]
        row["num_seqs"] = float(sum(chosen_data.get("nz_shifted_loss_weights", [0])) +
                                sum(rejected_data.get("nz_shifted_loss_weights", [0])))

        # Sentinel NaN: filled in phase 2 (compute_ref_logprobs) unless --no-ref-logps.
        # Training scripts auto-detect NaN and switch to online reference computation.
        row["chosen_ref_logp"] = [float("nan")]
        row["rejected_ref_logp"] = [float("nan")]

        outputs.append(row)

    job_print(job_id, "Chunk finish")
    return outputs


def _batch_to_tensor(row: dict) -> dict:
    """Convert a single row to model-compatible tensors."""
    keys = {
        "seqlens": torch.long,
        "nz_input_ids": torch.long,
        "nz_position_ids": torch.long,
        "nz_shifted_label_ids": torch.long,
        "nz_shifted_loss_weights": torch.bfloat16,
    }
    batch = {}
    for k, dtype in keys.items():
        arr = np.array(row[k], dtype=np.int32 if dtype == torch.long else np.float32)
        batch[k] = torch.from_numpy(arr).to(dtype)

    batch["cu_seqlens"] = torch.nn.functional.pad(
        batch["seqlens"].cumsum(-1, dtype=torch.int32), (1, 0)
    )
    batch["max_seqlen"] = batch["seqlens"].max().item()
    del batch["seqlens"]
    return batch


def compute_ref_logprobs(rows: list, args: DataArguments):
    """Phase 2: load the base model and compute reference log-probs for all pairs."""
    if args.no_ref_logps:
        return rows

    from ochat.config import MODEL_CONFIG_MAP

    print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Loading model for reference log-probs ...")

    model_config = MODEL_CONFIG_MAP[args.model_type]
    model = model_config.model_create_for_training(args.model_path, dtype=torch.bfloat16)
    model = model.to("cuda")
    model.eval()

    print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Computing reference log-probs for {len(rows)} pairs ...")

    with torch.inference_mode():
        for idx, row in enumerate(rows):
            for side in ("chosen", "rejected"):
                single = {k: np.array(row[f"{side}_{k}"]) for k in [
                    "seqlens", "nz_input_ids", "nz_position_ids",
                    "nz_shifted_label_ids", "nz_shifted_loss_weights",
                ]}
                tensor = _batch_to_tensor(single)
                tensor = {k: v.to("cuda") for k, v in tensor.items()}

                loss = model(**tensor, num_seq=1).loss
                if isinstance(loss, tuple):
                    loss, _ = loss

                # loss = sum(w_i * -log_p_i) with w_i=1 for response tokens
                # So ref_logp_sum = -loss
                row[f"{side}_ref_logp"] = [-loss.item()]

            if (idx + 1) % 500 == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Processed {idx + 1}/{len(rows)} pairs")
                torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return rows


def generate_split(rows: list, split_name: str, args: DataArguments):
    """Tokenize + compute ref log-probs + write parquet for one data split."""
    from ochat.config import MODEL_CONFIG_MAP

    metadata = {"model_type": args.model_type, "ref_logps_computed": not args.no_ref_logps}

    schema = [
        pyarrow.field("total_length", pyarrow.int32()),
        pyarrow.field("num_seqs", pyarrow.float32()),
    ]
    for side in ("chosen", "rejected"):
        for col in ("seqlens", "nz_input_ids", "nz_position_ids", "nz_shifted_label_ids", "nz_shifted_loss_weights"):
            dtype = pyarrow.float32() if col == "nz_shifted_loss_weights" else pyarrow.int32()
            schema.append(pyarrow.field(f"{side}_{col}", pyarrow.list_(dtype)))
        schema.append(pyarrow.field(f"{side}_images", pyarrow.list_(pyarrow.string())))
        schema.append(pyarrow.field(f"{side}_videos", pyarrow.list_(pyarrow.string())))
    schema.append(pyarrow.field("chosen_ref_logp", pyarrow.list_(pyarrow.float32())))
    schema.append(pyarrow.field("rejected_ref_logp", pyarrow.list_(pyarrow.float32())))

    schema = pyarrow.schema(schema, metadata={"metadata_json": orjson.dumps(metadata)})

    # Phase 1: tokenize in parallel (CPU)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        batches = list(enumerate(_split(rows, args.num_splits)))
        if not args.split_files:
            all_outputs = [None] * len(batches)
        for i in range(0, len(batches), args.max_jobs):
            subbatches = batches[i : i + args.max_jobs]
            handles = {
                executor.submit(convert_conversation_batch, job_id=job_id, batch=batch, args=args): job_id
                for job_id, batch in subbatches
            }
            for handle in concurrent.futures.as_completed(handles):
                job_id = handles.pop(handle)
                output = handle.result()
                if args.split_files:
                    # Write immediately (no ref logprobs computation in split mode)
                    parquet.write_table(
                        pyarrow.Table.from_pylist(output, schema=schema),
                        f"{args.out_prefix}.{split_name}.part{job_id:03d}.parquet",
                    )
                    job_print(job_id, "Write part file is done")
                else:
                    all_outputs[job_id] = output
                    job_print(job_id, "Collect result is done")
                gc.collect()

    if not args.split_files:
        all_rows = [row for output in all_outputs for row in output]

        # Phase 2: compute reference log-probs (GPU)
        all_rows = compute_ref_logprobs(all_rows, args)

        # Write
        print(f'[{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}] Write table to disk ...')
        parquet.write_table(
            pyarrow.Table.from_pylist(all_rows, schema=schema),
            f"{args.out_prefix}.{split_name}.parquet",
        )
        print(f'[{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}] Write finish')


def generate_dataset(args: DataArguments):
    # Load conversations
    lines = []
    for filename in args.in_files:
        with open(filename, "rt") as f:
            lines.extend(f.readlines())

    # Train-test split
    random.seed(args.seed)
    random.shuffle(lines)
    eval_num = int(args.eval_ratio * len(lines))

    train_lines = lines[eval_num:]
    eval_lines = lines[:eval_num]

    generate_split(train_lines, "train", args)
    if eval_num > 0:
        generate_split(eval_lines, "eval", args)


if __name__ == "__main__":
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
    parser.add_argument("--no-ref-logps", action="store_true", help="Skip reference log-prob computation (for debugging or if pre-computed elsewhere)")
    args = parser.parse_args()

    args = DataArguments(**vars(args))

    generate_dataset(args)
