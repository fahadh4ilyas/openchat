"""
Generate training data based on conversations

Usage: python -m ochat.data.generate_data --in-file sharegpt_gpt4.jsonl --tokenizer-name HF_REPO_NAME --out-dir .
"""

import concurrent.futures
from typing import List, Optional
import argparse
import os
import random

from pydantic import BaseModel, Field

import concurrent
import orjson
import pyarrow
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
    ignore_last_token: bool = Field(False)
    max_workers: Optional[int] = Field(None)


PAD_TOKEN_ID = 0


def _split(a: list, n: int):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def truncate_trailing_zero_weighted(tokens: list, weights: list):
    non_zero_index = len(weights) - 1
    while non_zero_index >= 0 and weights[non_zero_index] == 0:
        non_zero_index -= 1

    return tokens[:non_zero_index + 1], weights[:non_zero_index + 1]


def add_single_conv(output: dict, tokens: list, weights: list, args: DataArguments):
    # truncate trailing zero weighted tokens
    tokens, weights = truncate_trailing_zero_weighted(tokens, weights)
    if not tokens:
        return

    # labels
    LABEL_PAD_TOKEN_ID = args.ignore_index if args.ignore_index != PAD_TOKEN_ID else PAD_TOKEN_ID
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
    addition = -(-length // args.data_length_multiple_of) * args.data_length_multiple_of - length
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

        "nz_shifted_label_ids":    labels,
        "nz_shifted_loss_weights": weights
    }
    results["num_seqs"] = sum(results["nz_shifted_loss_weights"])

    for k, v in results.items():
        output[k].append(v)


def convert_conversation_batch(batch: list, schema: pyarrow.Schema, args: DataArguments):
    from ochat.config import MODEL_CONFIG_MAP, Conversation, PretokenizedConversation

    # Tokenization
    model_config = MODEL_CONFIG_MAP[args.model_type]
    tokenizer = model_config.model_tokenizer_create(args.model_path)
    conv_template = model_config.conversation_template(tokenizer=tokenizer)

    # Decode data
    print ("Decoding JSON ...")
    if args.pretokenized_in_files:
        batch = [PretokenizedConversation(**orjson.loads(json_line)) for json_line in batch]
        tokens_list = [b.input_ids for b in batch]
        weights_list = [b.loss_weights for b in batch]
    else:
        batch = [Conversation(**orjson.loads(json_line)) for json_line in batch]

        # Tokenize
        print ("Tokenizing ...")
        tokens_list = []
        weights_list = []
        if len(batch) > 0:
            tokens_list, weights_list = conv_template.tokenize_conversations(
                batch,
                inference=False,
                seq_level_weight=args.per_sequence_loss,
                force_eos_token=args.force_eos_token,
                eos_final=args.eos_final)

    # Generate data
    print ("Generating ...")
    max_context = args.max_seq_length or model_config.model_max_context

    outputs = {k: [] for k in schema.names}
    for tokens, weights in zip(tokens_list, weights_list):
        assert len(tokens) == len(weights)

        # Truncate to specified tokens
        tokens  = tokens[:max_context]
        weights = weights[:max_context]

        # Add to results
        add_single_conv(outputs, tokens, weights, args)

    print ("Chunk finish")

    return pyarrow.Table.from_pydict(outputs, schema=schema)


def generate_split(conversations: list, split_name: str, args: DataArguments):
    # schema
    metadata = {
        "model_type": args.model_type
    }
    schema = [
        pyarrow.field("total_length", pyarrow.int32()),
        pyarrow.field("num_seqs", pyarrow.float32()),

        pyarrow.field(f"seqlens", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_input_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_position_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_shifted_label_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_shifted_loss_weights", pyarrow.list_(pyarrow.float32()))
    ]

    schema = pyarrow.schema(schema, metadata={"metadata_json": orjson.dumps(metadata)})

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        handles = [executor.submit(convert_conversation_batch,
            batch=batch,
            schema=schema,
            args=args
        ) for batch in _split(conversations, executor._max_workers)]

    # write
    parquet.write_table(pyarrow.concat_tables([handle.result() for handle in handles]), f"{args.out_prefix}.{split_name}.parquet")


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
    eval_conversations  = conversations[:eval_num]

    generate_split(train_conversations, "train", args)
    if eval_num > 0:
        generate_split(eval_conversations, "eval", args)


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
    parser.add_argument("--pretokenized-in-files", action="store_true")
    parser.add_argument("--ignore-last-token", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    args = DataArguments(**vars(args))

    generate_dataset(args)
