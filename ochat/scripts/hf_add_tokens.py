import argparse

import transformers
import torch


def hf_add_tokens(model_path, output_dir, added_special_tokens, added_tokens):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                              low_cpu_mem_usage=True,
                                                              torch_dtype=torch.bfloat16)
    # Add tokens (tokenizer)
    tokenizer.add_tokens(added_special_tokens, special_tokens=True)
    tokenizer.add_tokens(added_tokens)

    # Add tokens (embedding)
    model.resize_token_embedddings(len(tokenizer))

    # Save
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        help="Location of Mistral model, or HuggingFace repo ID",
    )
    parser.add_argument(
        "--output-dir",
        help="Location to write resulting model and tokenizer",
    )
    parser.add_argument(
        "--added-special-tokens",
        type=str,
        nargs="+",
        help="Special token list to add"
    )
    parser.add_argument(
        "--added-tokens",
        type=str,
        nargs="*",
        help="Token list to add"
    )

    hf_add_tokens(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()