"""
Print available model types and their properties.

Usage:
    python -m ochat.config.list_models                  # list all
    python -m ochat.config.list_models --model-type llama  # single model
"""

import argparse

from ochat.config._model_config_data import MODEL_CONFIG_DATA


def _tokenizer_type(has_processor: bool) -> str:
    return "processor" if has_processor else "tokenizer"


def _print_row(name, max_context, has_processor, model_class, conv_type):
    print(
        f"{name:<30} "
        f"{model_class:<35} "
        f"{_tokenizer_type(has_processor):<9} "
        f"{conv_type:<10} "
        f"{max_context}"
    )


def main():
    parser = argparse.ArgumentParser(description="List available model types.")
    parser.add_argument("--model-type", type=str, default=None,
                        help="Show only this model type (omit for all)")
    args = parser.parse_args()

    # Build lookup
    data_by_name = {d[0]: d for d in MODEL_CONFIG_DATA}

    if args.model_type:
        if args.model_type not in data_by_name:
            available = ", ".join(sorted(data_by_name))
            print(f"Unknown model type: {args.model_type}")
            print(f"Available: {available}")
            return
        entry = data_by_name[args.model_type]
        name, max_context, has_processor, model_class, conv_type, _kw = entry[:6]
        print(f"model_type:       {name}")
        print(f"model_class:      {model_class}")
        print(f"tokenizer_type:   {_tokenizer_type(has_processor)}")
        print(f"conversation:     {conv_type}")
        print(f"max_context:      {max_context}")
    else:
        print(f"{'Model Type':<30} {'Class':<35} {'T/P':<9} {'Conv':<10} {'Context'}")
        print("-" * 110)
        for entry in MODEL_CONFIG_DATA:
            name, max_context, has_processor, model_class, conv_type, _kw = entry[:6]
            _print_row(name, max_context, has_processor, model_class, conv_type)


if __name__ == "__main__":
    main()
