import importlib
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

# Mapping from MODEL_CONFIG_DATA class_name → (module_path, attr_name_in_module).
# Many modules define a base class (e.g. LlamaForCausalLM) while
# ochat.models.__init__ re-exports it under an alias (e.g. LlamaSplitForCausalLM).
# This map records the actual attribute name to look up in each module.
_MODEL_CLASS_TO_MODULE = {
    "LlamaForCausalLM":                    ("ochat.models.unpadded_llama",           "LlamaForCausalLM"),
    "LlamaSplitForCausalLM":               ("ochat.models.unpadded_llama_split",     "LlamaForCausalLM"),
    "LlamaLongForCausalLM":                ("ochat.models.unpadded_llama_long",      "LlamaForCausalLM"),
    "LlamaRingForCausalLM":                ("ochat.models.unpadded_llama_ring",      "LlamaForCausalLM"),
    "LlamaLongRingForCausalLM":            ("ochat.models.unpadded_llama_long_ring", "LlamaForCausalLM"),
    "MistralForCausalLM":                  ("ochat.models.unpadded_mistral",         "MistralForCausalLM"),
    "MistralSplitForCausalLM":             ("ochat.models.unpadded_mistral_split",   "MistralForCausalLM"),
    "MistralLongForCausalLM":              ("ochat.models.unpadded_mistral_long",    "MistralForCausalLM"),
    "MistralRingForCausalLM":              ("ochat.models.unpadded_mistral_ring",    "MistralForCausalLM"),
    "MixtralForCausalLM":                  ("ochat.models.unpadded_mixtral",         "MixtralForCausalLM"),
    "MixtralRingForCausalLM":              ("ochat.models.unpadded_mixtral_ring",    "MixtralForCausalLM"),
    "GemmaForCausalLM":                    ("ochat.models.unpadded_gemma",           "GemmaForCausalLM"),
    "Gemma2ForCausalLM":                   ("ochat.models.unpadded_gemma2",          "Gemma2ForCausalLM"),
    "GemmaRingForCausalLM":                ("ochat.models.unpadded_gemma_ring",      "GemmaForCausalLM"),
    "PhiForCausalLM":                      ("ochat.models.unpadded_phi",             "PhiForCausalLM"),
    "OriPhiForCausalLM":                   ("ochat.models.unpadded_phi_ori",         "PhiForCausalLM"),
    "Qwen2ForCausalLM":                    ("ochat.models.unpadded_qwen2",           "Qwen2ForCausalLM"),
    "Qwen2RingForCausalLM":                ("ochat.models.unpadded_qwen2_ring",      "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM":                    ("ochat.models.unpadded_qwen3",           "Qwen3ForCausalLM"),
    "Qwen3RingForCausalLM":                ("ochat.models.unpadded_qwen3_ring",      "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM":                 ("ochat.models.unpadded_qwen3_moe",       "Qwen3MoeForCausalLM"),
    "Qwen3MoeRingForCausalLM":             ("ochat.models.unpadded_qwen3_moe_ring",  "Qwen3MoeForCausalLM"),
    "Qwen3_5ForConditionalGeneration":     ("ochat.models.unpadded_qwen3_5",         "Qwen3_5ForConditionalGeneration"),
    "Qwen3_5MoeForConditionalGeneration":  ("ochat.models.unpadded_qwen3_5_moe",     "Qwen3_5MoeForConditionalGeneration"),
    "DeepseekV2ForCausalLM":               ("ochat.models.unpadded_deepseekv2",      "DeepseekV2ForCausalLM"),
}

# O(1) lookup: model name → raw MODEL_CONFIG_DATA entry
_KEY_TO_ENTRY = {entry[0]: entry for entry in MODEL_CONFIG_DATA}
_ALL_KEYS = tuple(_KEY_TO_ENTRY.keys())


def _v3_2_role_prefix(from_role: str, condition: str):
    return f"{condition} {_V3_2_PREFIXES.get(from_role, from_role+':')}".strip()


def _chatml_role_prefix(from_role: str, condition: str, model: str):
    return f"{condition} {PREFIXES.get(model, {}).get(from_role, from_role)}".strip()


def _import_model_class(class_name: str):
    """Lazily import a single model class from its submodule."""
    module_path, attr_name = _MODEL_CLASS_TO_MODULE[class_name]
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _build_config_for_entry(entry):
    """Build a single ModelConfig from one MODEL_CONFIG_DATA entry.

    Only the one needed model submodule is imported; torch / transformers
    are imported on first call and cached by Python's import system.
    """
    import torch
    import transformers

    from ochat.config.model_config import ModelConfig
    from ochat.config.conversation_template import (
        ConversationTemplate,
        ChatMLConversationTemplate,
        DeepseekConversationTemplate,
    )

    name, max_context, has_processor, model_class_name, conv_type, kwargs = entry[:6]
    model_kwargs = entry[6] if len(entry) > 6 else {}

    # Tokenizer
    use_fast = True if name == "deepseekv2" else False
    if has_processor:
        tokenizer_create = partial(transformers.AutoProcessor.from_pretrained)
    else:
        tokenizer_create = partial(transformers.AutoTokenizer.from_pretrained, use_fast=use_fast)

    # Model class — lazy, imports only the one needed submodule
    model_cls = _import_model_class(model_class_name)
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

    return ModelConfig(
        model_max_context=max_context,
        model_tokenizer_create=tokenizer_create,
        model_create_for_training=model_create,
        model_has_processor=has_processor,
        conversation_template=template,
    )


class _LazyConfigMap:
    """Lazy wrapper around MODEL_CONFIG_DATA.

    - keys(), __getitem__, __contains__: no model imports (single import for __getitem__).
    - values(), items(): import ALL models at once.
    - __iter__: imports models one by one as iteration proceeds.
    """
    def __init__(self):
        self._map = None       # name → ModelConfig cache
        self._all_built = False

    # ── lightweight access (no model imports) ──────────────────────

    def __contains__(self, key):
        return key in _KEY_TO_ENTRY

    def keys(self):
        return _ALL_KEYS

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    # ── single-model lazy access ───────────────────────────────────

    def __getitem__(self, key):
        if self._all_built:
            return self._map[key]
        if self._map is None:
            self._map = {}
        if key not in self._map:
            entry = _KEY_TO_ENTRY[key]
            self._map[key] = _build_config_for_entry(entry)
        return self._map[key]

    # ── iterator: builds models one by one ────────────────────────

    def __iter__(self):
        if self._all_built:
            return iter(self._map)
        return self._lazy_iter()

    def _lazy_iter(self):
        if self._map is None:
            self._map = {}
        for entry in MODEL_CONFIG_DATA:
            name = entry[0]
            if name not in self._map:
                self._map[name] = _build_config_for_entry(entry)
            yield name

    # ── bulk access: imports ALL models ────────────────────────────

    def values(self):
        self._build_all()
        return self._map.values()

    def items(self):
        self._build_all()
        return self._map.items()

    def _build_all(self):
        if self._all_built:
            return
        if self._map is None:
            self._map = {}
        for entry in MODEL_CONFIG_DATA:
            name = entry[0]
            if name not in self._map:
                self._map[name] = _build_config_for_entry(entry)
        self._all_built = True


MODEL_CONFIG_MAP = _LazyConfigMap()
