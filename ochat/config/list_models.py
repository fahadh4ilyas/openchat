"""
Print all available model types and their properties.

Usage: python -m ochat.config.list_models
"""

import sys

from ochat.config import MODEL_CONFIG_MAP
from ochat.config.conversation_template import (
    ConversationTemplate,
    ChatMLConversationTemplate,
    DeepseekConversationTemplate,
)


def _conv_type(template_partial) -> str:
    """Detect conversation template type."""
    func = template_partial.func if hasattr(template_partial, "func") else type(template_partial).__name__
    if issubclass(func, ChatMLConversationTemplate):
        return "chatml"
    elif issubclass(func, DeepseekConversationTemplate):
        return "deepseek"
    elif issubclass(func, ConversationTemplate):
        return "openchat"
    return "unknown"


def _model_class(config) -> str:
    """Extract model class name from model_create_for_training partial."""
    func = config.model_create_for_training.func
    if hasattr(func, "__self__"):
        return func.__self__.__name__
    return func.__name__


def _tokenizer_type(config) -> str:
    """Return 'tokenizer' or 'processor'."""
    return "processor" if config.model_has_processor else "tokenizer"


def main():
    print(f"{'Model Type':<30} {'Class':<35} {'T/P':<9} {'Conv':<10} {'Context'}")
    print("-" * 110)

    for name in sorted(MODEL_CONFIG_MAP):
        config = MODEL_CONFIG_MAP[name]
        print(
            f"{name:<30} "
            f"{_model_class(config):<35} "
            f"{_tokenizer_type(config):<9} "
            f"{_conv_type(config.conversation_template):<10} "
            f"{config.model_max_context}"
        )


if __name__ == "__main__":
    main()
