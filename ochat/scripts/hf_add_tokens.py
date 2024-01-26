import argparse
import typing

import transformers

from pydantic import BaseModel, Field, validator

class Arguments(BaseModel):

    model_path: str = Field(...)
    save_path: str = Field(...)
    added_special_tokens: typing.List[str] = Field([])
    added_tokens: typing.List[str] = Field([])

    @validator('added_tokens')
    def validate_tokens(cls, value: typing.List[str], values: typing.Dict[str, typing.Any]) -> typing.List[str]:
        if len(value + values.get('added_special_tokens', [])) == 0:
            raise ValueError('At least one token added!')
        return value


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        help="Location of model, or HuggingFace repo ID",
    )
    parser.add_argument(
        "--output-dir",
        help="Location to write resulting model and tokenizer",
    )
    parser.add_argument(
        "--added-special-tokens",
        type=str,
        nargs="*",
        help="Special token list to add"
    )
    parser.add_argument(
        "--added-tokens",
        type=str,
        nargs="*",
        help="Token list to add"
    )

    args, _ = parser.parse_known_args()

    return args


def main(args: Arguments):

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    model: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(args.model_path,
                                                              low_cpu_mem_usage=True,
                                                              torch_dtype='auto')

    # Add tokens (tokenizer)
    tokenizer.add_tokens(args.added_special_tokens, special_tokens=True)
    tokenizer.add_special_tokens({'additional_special_tokens': list(set(tokenizer.special_tokens_map_extended.get('additional_special_tokens', []) + args.added_special_tokens))}, replace_additional_special_tokens=False)
    tokenizer.add_tokens(args.added_tokens)

    # Add tokens (embedding)
    model.resize_token_embeddings(len(tokenizer))

    # Save
    tokenizer.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)


if __name__ == "__main__":

    args = parse_args()
    args = Arguments(**vars(args))
    main(args)