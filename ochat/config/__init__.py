from functools import partial

from ochat.config._model_config_data import MODEL_CONFIG_DATA


_V3_2_PREFIXES = {
    # OpenAI mapping
    "user": "User:",
    "assistant": "Assistant:",
}

PREFIXES = {
    "gemma": {"user": "user", "assistant": "model"},
    "gemma2": {"user": "user", "assistant": "model"},
}


def _v3_2_role_prefix(from_role: str, condition: str):
    return f"{condition} {_V3_2_PREFIXES.get(from_role, from_role+':')}".strip()


def _chatml_role_prefix(from_role: str, condition: str, model: str):
    return f"{condition} {PREFIXES.get(model, {}).get(from_role, from_role)}".strip()


def _build_config_map():
    """Build MODEL_CONFIG_MAP from lightweight data (lazy — only runs on first access)."""
    import torch
    import transformers
    import ochat.models

    from ochat.config.model_config import ModelConfig
    from ochat.config.conversation_template import (
        ConversationTemplate,
        ChatMLConversationTemplate,
        DeepseekConversationTemplate,
    )

    config_map = {}
    for entry in MODEL_CONFIG_DATA:
        name, max_context, has_processor, model_class_name, conv_type, kwargs = entry[:6]
        model_kwargs = entry[6] if len(entry) > 6 else {}
        # Tokenizer
        use_fast = True if name == "deepseekv2" else False
        if has_processor:
            tokenizer_create = partial(transformers.AutoProcessor.from_pretrained)
        else:
            tokenizer_create = partial(transformers.AutoTokenizer.from_pretrained, use_fast=use_fast)

        # Model class
        model_cls = getattr(ochat.models, model_class_name)
        model_create = partial(model_cls.from_pretrained, dtype=torch.bfloat16, **model_kwargs)

        # Conversation template
        if conv_type == "openchat":
            template = partial(ConversationTemplate, role_prefix=_v3_2_role_prefix, **kwargs)
        elif conv_type == "chatml":
            template = partial(ChatMLConversationTemplate, role_prefix=_chatml_role_prefix, **kwargs)
        elif conv_type == "deepseek":
            template = partial(DeepseekConversationTemplate, **kwargs)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        config_map[name] = ModelConfig(
            model_max_context=max_context,
            model_tokenizer_create=tokenizer_create,
            model_create_for_training=model_create,
            model_has_processor=has_processor,
            conversation_template=template,
        )

    return config_map


class _LazyConfigMap:
    """Lazy wrapper — only builds MODEL_CONFIG_MAP on first access."""
    def __init__(self):
        self._map = None

    def _ensure(self):
        if self._map is None:
            self._map = _build_config_map()

    def __getitem__(self, key):
        self._ensure()
        return self._map[key]

    def __contains__(self, key):
        self._ensure()
        return key in self._map

    def __iter__(self):
        self._ensure()
        return iter(self._map)

    def keys(self):
        self._ensure()
        return self._map.keys()

    def values(self):
        self._ensure()
        return self._map.values()

    def items(self):
        self._ensure()
        return self._map.items()

    def get(self, key, default=None):
        self._ensure()
        return self._map.get(key, default)


MODEL_CONFIG_MAP = _LazyConfigMap()
