"""
Generate training data based on conversations

Usage: python -m ochat.data.generate_dataset --in-files sharegpt_gpt4.jsonl --model-type MODEL_TYPE --model-path BASE_REPO --out-prefix .
       python -m ochat.data.generate_dataset --train-type dpo --in-files data_dpo.jsonl --model-type MODEL_TYPE --model-path BASE_REPO --out-prefix .
"""

import gc
import concurrent.futures
from typing import List, Optional
import argparse
from datetime import datetime
import random

from pydantic import BaseModel, Field, field_validator as validator, ValidationInfo

import concurrent
import orjson
import pyarrow
import numpy as np
from pyarrow import parquet


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
    pretokenized_in_files: bool = Field(False)
    pretraining_data: bool = Field(False)
    ignore_last_token: bool = Field(False)
    separate_think: bool = Field(False)
    max_workers: Optional[int] = Field(None)
    max_jobs: int = Field(10)
    split_files: bool = Field(False)
    num_splits: int = Field(10)
    kto: bool = Field(False)
    ref_logps: bool = Field(False)

    @validator("max_jobs")
    def check_max_jobs(cls, v, info: ValidationInfo):
        if info.data['max_workers'] is not None and v > info.data['max_workers']:
            raise ValueError("max_jobs cannot be greater than max_workers")
        return v


PAD_TOKEN_ID = 0


def job_print(job_id: int, *args, **kwargs):
    print(
        f'[{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}] [JOB ID: {job_id}]',
        *args,
        **kwargs,
    )


def _split(a: list, n: int):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def truncate_trailing_zero_weighted(tokens: list, weights: list):
    non_zero_index = len(weights) - 1
    while non_zero_index >= 0 and weights[non_zero_index] == 0:
        non_zero_index -= 1

    return tokens[: non_zero_index + 1], weights[: non_zero_index + 1]


def add_single_conv(outputs: list, tokens: list, weights: list, images: list, videos: list, args: DataArguments,
                    label: Optional[bool] = None, ref_logp: Optional[float] = None):
    # truncate trailing zero weighted tokens
    tokens, weights = truncate_trailing_zero_weighted(tokens, weights)
    if not tokens:
        return

    # labels
    LABEL_PAD_TOKEN_ID = (
        args.ignore_index if args.ignore_index != PAD_TOKEN_ID else PAD_TOKEN_ID
    )
    labels = [(t if w != 0 else LABEL_PAD_TOKEN_ID) for t, w in zip(tokens, weights)]

    # Shift data
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

    # Pad data
    addition = (
        -(-length // args.data_length_multiple_of) * args.data_length_multiple_of
        - length
    )
    if addition > 0:
        tokens.extend(last_token + [PAD_TOKEN_ID] * (addition - 1))
        weights.extend([0.0] * addition)
        labels.extend([LABEL_PAD_TOKEN_ID] * addition)
    length = len(tokens)

    # populate results
    results = {
        "total_length": length,
        "seqlens": [length],
        "nz_input_ids": tokens,
        "nz_position_ids": list(range(length)),
        "nz_shifted_label_ids": labels,
        "nz_shifted_loss_weights": weights,
        "images": images,
        "videos": videos,
    }
    results["num_seqs"] = sum(results["nz_shifted_loss_weights"])

    if label is not None:
        results["label"] = label
    if ref_logp is not None:
        results["ref_logp"] = float(ref_logp)

    outputs.append(results)


def convert_conversation_batch(job_id: int, batch: list, args: DataArguments):
    from ochat.config import MODEL_CONFIG_MAP
    from ochat.config.conversation_template import Conversation, PretokenizedConversation, PretrainingText

    # Tokenization
    model_config = MODEL_CONFIG_MAP[args.model_type]
    tokenizer = model_config.model_tokenizer_create(args.model_path)

    image_processor = None
    video_processor = None
    if model_config.model_has_processor:
        if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
            conv_template = model_config.conversation_template(tokenizer=tokenizer.tokenizer)
        else:
            conv_template = model_config.conversation_template(tokenizer=tokenizer)
        if hasattr(tokenizer, "image_processor") and tokenizer.image_processor is not None:
            image_processor = tokenizer.image_processor
        if hasattr(tokenizer, "video_processor") and tokenizer.video_processor is not None:
            video_processor = tokenizer.video_processor
    else:
        conv_template = model_config.conversation_template(tokenizer=tokenizer)

    # Decode data
    job_print(job_id, "Decoding JSON ...")
    if args.pretokenized_in_files:
        curr_batch: List[PretokenizedConversation] = [
            PretokenizedConversation(**orjson.loads(json_line)) for json_line in batch
        ]
        tokens_list = [b.input_ids for b in curr_batch]
        weights_list = [b.loss_weights for b in curr_batch]
    elif args.pretraining_data:
        curr_batch: List[PretrainingText] = [
            PretrainingText(**orjson.loads(json_line)) for json_line in batch
        ]
        all_text = [b.text for b in curr_batch]
        text_mapping = dict(zip(all_text, conv_template._safe_tokenize(all_text)))
        tokens_list = []
        weights_list = []
        for b in curr_batch:
            tokens = []

            tokens.extend(conv_template.bos_tokens_)
            token_msg = text_mapping[b.text]
            tokens.extend(token_msg)
            if (args.force_eos_token or args.eos_final) and tokens[-1] != conv_template.eos_tokens_[0]:
                tokens.extend(conv_template.eos_tokens_)
            if args.per_sequence_loss:
                weights = [b.weight / len(tokens)] * len(tokens)
            else:
                weights = [b.weight] * len(tokens)

            tokens_list.append(tokens)
            weights_list.append(weights)
    else:
        curr_batch: List[Conversation] = [Conversation(**orjson.loads(json_line)) for json_line in batch]

        # Tokenize
        job_print(job_id, "Tokenizing ...")
        tokens_list = []
        weights_list = []
        if len(curr_batch) > 0:
            tokens_list, weights_list = conv_template.tokenize_conversations(
                curr_batch,
                inference=False,
                seq_level_weight=args.per_sequence_loss,
                force_eos_token=args.force_eos_token,
                eos_final=args.eos_final,
                separate_think=args.separate_think,
            )
    
    has_image = any([bool(b.images) for b in curr_batch])
    has_video = any([bool(b.videos) for b in curr_batch])
    images_list = [b.images if b.images else [] for b in curr_batch]
    videos_list = [b.videos if b.videos else [] for b in curr_batch]
    if has_image and image_processor is None:
        job_print(job_id, "Warning: The tokenizer does not have an image processor but the data contains images. The images will be ignored.")
    if has_video and video_processor is None:
            job_print(job_id, "Warning: The tokenizer does not have a video processor but the data contains videos. The videos will be ignored.")

    # Generate data
    job_print(job_id, "Generating ...")
    max_context = args.max_seq_length or model_config.model_max_context

    outputs = []
    for i, (tokens, weights, images, videos) in enumerate(zip(tokens_list, weights_list, images_list, videos_list)):
        assert len(tokens) == len(weights)

        # Truncate to specified tokens
        tokens = tokens[:max_context]
        weights = weights[:max_context]

        kwargs = {}
        if args.kto:
            kwargs["label"] = curr_batch[i].label
            kwargs["ref_logp"] = float("nan")  # Sentinel NaN: filled by compute_ref_logprobs if --ref-logps

        # Add to results
        add_single_conv(outputs, tokens, weights, images, videos, args, **kwargs)

    job_print(job_id, "Chunk finish")

    return outputs


def _batch_to_tensor(row: dict) -> dict:
    """Convert a single row dict to model-compatible tensors."""
    import torch
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


def _compute_kto_ref_logprobs(rows: list, args: DataArguments):
    """Compute reference log-probs for KTO data by running each example through the base model.

    Each row gets a scalar ref_logp = log p_ref(response | prompt).
    Uses the SFT-style forward (num_seq=1, loss is CE with loss_weights).
    ref_logp = -loss (since loss = -sum(w_i * log p_i) with w_i=1 for response tokens).
    """
    import torch
    from ochat.config import MODEL_CONFIG_MAP

    print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Loading model for KTO reference log-probs ...")

    model_config = MODEL_CONFIG_MAP[args.model_type]
    model = model_config.model_create_for_training(args.model_path, dtype=torch.bfloat16)
    model = model.to("cuda")
    model.eval()

    print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Computing reference log-probs for {len(rows)} examples ...")

    with torch.inference_mode():
        for idx, row in enumerate(rows):
            single = {k: np.array(row[k]) for k in [
                "seqlens", "nz_input_ids", "nz_position_ids",
                "nz_shifted_label_ids", "nz_shifted_loss_weights",
            ]}
            tensor = _batch_to_tensor(single)
            tensor = {k: v.to("cuda") for k, v in tensor.items()}

            loss = model(**tensor, num_seq=1).loss
            if isinstance(loss, tuple):
                loss, _ = loss

            row["ref_logp"] = -loss.item()

            if (idx + 1) % 500 == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] Processed {idx + 1}/{len(rows)} examples")
                torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return rows


def generate_split(conversations: list, split_name: str, args: DataArguments):
    # schema
    metadata = {"model_type": args.model_type}
    if args.kto:
        metadata["ref_logps_computed"] = args.ref_logps
    schema = [
        pyarrow.field("total_length", pyarrow.int32()),
        pyarrow.field("num_seqs", pyarrow.float32()),
        pyarrow.field(f"seqlens", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_input_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_position_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_shifted_label_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_shifted_loss_weights", pyarrow.list_(pyarrow.float32())),
        pyarrow.field(f"images", pyarrow.list_(pyarrow.string())),
        pyarrow.field(f"videos", pyarrow.list_(pyarrow.string()))
    ]
    if args.kto:
        schema += [
            pyarrow.field("label", pyarrow.bool_()),
            pyarrow.field("ref_logp", pyarrow.float32()),
        ]

    schema = pyarrow.schema(schema, metadata={"metadata_json": orjson.dumps(metadata)})

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        batches = list(enumerate(_split(conversations, args.num_splits)))
        if not args.split_files:
            outputs = [None] * len(batches)
        for i in range(0, len(batches), args.max_jobs):
            subbatches = batches[i:i+args.max_jobs]
            handles = {
                executor.submit(
                    convert_conversation_batch, job_id=job_id, batch=batch, args=args
                ): job_id
                for job_id, batch in subbatches
            }

            # Collecting
            for handle in concurrent.futures.as_completed(handles):
                job_id = handles.pop(handle)
                output = handle.result()
                if args.split_files:
                    # write immediately
                    parquet.write_table(
                        pyarrow.Table.from_pylist(output, schema=schema),
                        f"{args.out_prefix}.{split_name}.part{job_id:03d}.parquet",
                    )
                    job_print(job_id, "Write part file is done")
                else:
                    outputs[job_id] = output
                    job_print(job_id, "Collect result is done")
                gc.collect()
        if not args.split_files:
            outputs = [d for output in outputs for d in output]

    # Compute reference log-probs for KTO (if --kto --ref-logps)
    if args.kto and args.ref_logps:
        outputs = _compute_kto_ref_logprobs(outputs, args)

    # write
    if not args.split_files:
        print(f'[{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}] Write table to disk ...')
        parquet.write_table(
            pyarrow.Table.from_pylist(outputs, schema=schema),
            f"{args.out_prefix}.{split_name}.parquet",
        )
        print(f'[{datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}] Write finish')


def generate_dataset(args: DataArguments):
    # Load conversations
    conversations = []
    for filename in args.in_files:
        with open(filename, "rt") as f:
            conversations.extend(f.readlines())

    # Train-test split
    random.seed(args.seed)
    random.shuffle(conversations)
    eval_num = int(args.eval_ratio * len(conversations))

    train_conversations = conversations[eval_num:]
    eval_conversations = conversations[:eval_num]

    generate_split(train_conversations, "train", args)
    if eval_num > 0:
        generate_split(eval_conversations, "eval", args)


def _delegate_to_train_type(train_type: str):
    """Remove --train-type from sys.argv and delegate to generate_dataset_<train_type>.main()."""
    import sys
    import importlib

    cleaned_argv = []
    skip_next = False
    for arg in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--train-type":
            skip_next = True
            continue
        if arg.startswith("--train-type="):
            continue
        cleaned_argv.append(arg)
    sys.argv = cleaned_argv

    module_name = f"ochat.data.generate_{train_type}_dataset"
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"Unknown train type '{train_type}': no module '{module_name}' found."
        )
    module.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--in-files", type=str, nargs="+", required=True)
    parser.add_argument("--out-prefix", type=str, required=True)

    parser.add_argument("--per-sequence-loss", action="store_true", help="If true, the total weight of the sequence is divided by the number of tokens in the sequence. If false, the total weight is used as is.")
    parser.add_argument("--force-eos-token", action="store_true", help="If true, the EOS token is added at the end of every sequence with non zero weight.")
    parser.add_argument("--eos-final", action="store_true", help="If true, the EOS token is added to the very end of the sequence, even if the last token is not a EOS token.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="If specified, the sequence is truncated to this length. If not specified, the maximum context length of the model is used.")
    parser.add_argument("--ignore-index", type=int, default=0, help="The value used to pad the labels. If 0, the PAD_TOKEN_ID is used which is 0.")
    parser.add_argument("--seed", type=int, default=42, help="The seed used to shuffle the data.")
    parser.add_argument("--eval-ratio", type=float, default=0.0, help="The ratio of the data used for evaluation. If 0.0, no evaluation data is generated.")
    parser.add_argument("--data-length-multiple-of", type=int, default=1, help="The length of the data is a multiple of this value. If 1, no padding is done.")
    parser.add_argument("--pretokenized-in-files", action="store_true", help="If true, the input files are pretokenized. If false, the input files are not pretokenized.")
    parser.add_argument("--pretraining-data", action="store_true", help="If true, the input files are pretraining data. If false, the input files are not pretraining data.")
    parser.add_argument("--ignore-last-token", action="store_true", help="If true, the last token of sequence is ignored. If false, the labels will be padded to match the input sequence.")
    parser.add_argument("--separate-think", action="store_true", help="If true, if the sequence contains a think token, the sequence will be separated into multiple sequences with thinking only on the end of the sequence. If false, the sequence will be treated as a single sequence.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=10)
    parser.add_argument("--num-splits", type=int, default=10, help="Number of split jobs to create.")
    parser.add_argument("--split-files", action="store_true", help="If true, the input files are split into multiple files for processing based on num_splits. If false, the input files are processed as a single file.")
    parser.add_argument("--kto", action="store_true", help="KTO mode: add label and ref_logp columns for KTO training.")
    parser.add_argument("--ref-logps", action="store_true", help="Compute reference log-probs during preprocessing (requires GPU, only meaningful with --kto).")
    parser.add_argument("--train-type", type=str, default="sft",
                        help="Training type: sft, dpo, orpo, or kto. "
                             "If not sft, delegates to generate_dataset_<train_type>.py.")
    args = parser.parse_args()

    if args.train_type != "sft":
        _delegate_to_train_type(args.train_type)
        import sys
        sys.exit(0)

    args = DataArguments(**vars(args))

    generate_dataset(args)
