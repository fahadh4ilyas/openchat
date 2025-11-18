from functools import partial

import torch
import transformers

from ochat.config.model_config import ModelConfig
from ochat.config.conversation_template import (
    Conversation,
    PretokenizedConversation,
    PretrainingText,
    ConversationTemplate,
    ChatMLConversationTemplate,
    DeepseekConversationTemplate
)
import ochat.models


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


MODEL_CONFIG_MAP = {
    # OpenChat V3.2
    "llama": ModelConfig(
        # Model
        model_max_context=4096,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4",
        ),
    ),
    "llamaSplit": ModelConfig(
        # Model
        model_max_context=4096,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaSplitForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4",
        ),
    ),
    "llamaLong": ModelConfig(
        # Model
        model_max_context=8 * 4096,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaLongForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4",
        ),
    ),
    "llamaRing": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4",
        ),
    ),
    "llama3": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="llama",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>",
            sep="\n\n",
            conv_sep="",
            inference_condition="GPT4",
        ),
    ),
    "llama3Ring": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="llama",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>",
            sep="\n\n",
            conv_sep="",
            inference_condition="GPT4",
        ),
    ),
    "llama3.1": ModelConfig(
        # Model
        model_max_context=2**17,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaLongForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="llama",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>",
            sep="\n\n",
            conv_sep="",
            inference_condition="GPT4",
        ),
    ),
    "llama3.1Ring": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaLongRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="llama",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>",
            sep="\n\n",
            conv_sep="",
            inference_condition="GPT4",
        ),
    ),
    "mistral": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MistralForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "mistralSplit": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MistralSplitForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "mistralLong": ModelConfig(
        # Model
        model_max_context=4 * 8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MistralLongForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "mistralRing": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MistralRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "mixtral": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MixtralForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "mixtralRing": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MixtralRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "phi": ModelConfig(
        # Model
        model_max_context=2048,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.PhiForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "phi_ori": ModelConfig(
        # Model
        model_max_context=2048,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.OriPhiForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "gemma": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.GemmaForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "gemmaRing": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.GemmaRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "gemma2": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Gemma2ForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "qwen2": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen2ForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "qwen2Ring": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen2RingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "qwen3": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3ForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "qwen3Ring": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3RingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "qwen3Moe": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3MoeForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "qwen3MoeRing": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3MoeRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ConversationTemplate,
            role_prefix=_v3_2_role_prefix,
            eot="<|end_of_turn|>",
            inference_condition="GPT4 Correct",
        ),
    ),
    "deepseekv2": ModelConfig(
        # Model
        model_max_context=163840,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=True
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.DeepseekV2ForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            DeepseekConversationTemplate,
            prompt_format={
                "user": "<｜{role}｜>{text}",
                "assistant": "<｜{role}｜>{text}<｜end▁of▁sentence｜>",
            },
        ),
    ),
    "llama_chatml": ModelConfig(
        # Model
        model_max_context=4096,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="llama",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4",
        ),
    ),
    "llamaSplit_chatml": ModelConfig(
        # Model
        model_max_context=4096,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaSplitForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="llama",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4",
        ),
    ),
    "llamaLong_chatml": ModelConfig(
        # Model
        model_max_context=8 * 4096,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaLongForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="llama",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4",
        ),
    ),
    "llamaRing_chatml": ModelConfig(
        # Model
        model_max_context=4096,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),
        model_create_for_training=partial(
            ochat.models.LlamaRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="llama",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4",
        ),
    ),
    "mistral_chatml": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MistralForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="mistral",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "mistralSplit_chatml": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MistralSplitForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="mistral",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "mistralLong_chatml": ModelConfig(
        # Model
        model_max_context=4 * 8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MistralLongForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="mistral",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "mistralRing_chatml": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MistralRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="mistral",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "mixtral_chatml": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MixtralForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="mixtral",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "mixtralRing_chatml": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MixtralRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="mixtral",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "phi_chatml": ModelConfig(
        # Model
        model_max_context=2048,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.PhiForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="phi",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "phi_ori_chatml": ModelConfig(
        # Model
        model_max_context=2048,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.OriPhiForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="phi",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "gemma_chatml": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.GemmaForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="gemma_chatml",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "gemmaRing_chatml": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.GemmaRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="gemma_chatml",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "gemma2_chatml": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Gemma2ForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="gemma2_chatml",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "gemma_instruct": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.GemmaForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="gemma",
            role_prefix=_chatml_role_prefix,
            prompt_format="<start_of_turn>{role}\n{text}<end_of_turn>",
            inference_condition="GPT4 correct",
        ),
    ),
    "gemmaRing_instruct": ModelConfig(
        # Model
        model_max_context=2**19,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.GemmaRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="gemma",
            role_prefix=_chatml_role_prefix,
            prompt_format="<start_of_turn>{role}\n{text}<end_of_turn>",
            inference_condition="GPT4 correct",
        ),
    ),
    "gemma2_instruct": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Gemma2ForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="gemma2",
            role_prefix=_chatml_role_prefix,
            prompt_format="<start_of_turn>{role}\n{text}<end_of_turn>",
            inference_condition="GPT4 correct",
        ),
    ),
    "qwen2_chatml": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen2ForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="qwen2",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "qwen2Ring_chatml": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen2RingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="qwen2",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "qwen3_chatml": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3ForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="qwen3",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "qwen3Ring_chatml": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3RingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="qwen3",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "qwen3Moe_chatml": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3MoeForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="qwen3_moe",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "qwen3MoeRing_chatml": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3MoeRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="qwen3_moe",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
            inference_condition="GPT4 correct",
        ),
    ),
    "qwen2_deepseek": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen2ForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            DeepseekConversationTemplate,
            prompt_format={
                "user": "<｜{role}｜>{text}",
                "assistant": "<｜{role}｜>{text}<｜end▁of▁sentence｜>",
            },
        ),
    ),
    "qwen2Ring_deepseek": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen2RingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            DeepseekConversationTemplate,
            prompt_format={
                "user": "<｜{role}｜>{text}",
                "assistant": "<｜{role}｜>{text}<｜end▁of▁sentence｜>",
            },
        ),
    ),
    "qwen3_deepseek": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3ForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            DeepseekConversationTemplate,
            prompt_format={
                "user": "<｜{role}｜>{text}",
                "assistant": "<｜{role}｜>{text}<｜end▁of▁sentence｜>",
            },
        ),
    ),
    "qwen3Ring_deepseek": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3RingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            DeepseekConversationTemplate,
            prompt_format={
                "user": "<｜{role}｜>{text}",
                "assistant": "<｜{role}｜>{text}<｜end▁of▁sentence｜>",
            },
        ),
    ),
    "qwen3Moe_deepseek": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3MoeForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            DeepseekConversationTemplate,
            prompt_format={
                "user": "<｜{role}｜>{text}",
                "assistant": "<｜{role}｜>{text}<｜end▁of▁sentence｜>",
            },
        ),
    ),
    "qwen3MoeRing_deepseek": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.Qwen3MoeRingForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        # Conversation Template
        conversation_template=partial(
            DeepseekConversationTemplate,
            prompt_format={
                "user": "<｜{role}｜>{text}",
                "assistant": "<｜{role}｜>{text}<｜end▁of▁sentence｜>",
            },
        ),
    ),
    "zephyr": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(
            transformers.AutoTokenizer.from_pretrained, use_fast=False
        ),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(
            ochat.models.MistralForCausalLM.from_pretrained, torch_dtype=torch.bfloat16
        ),
        # Conversation Template
        conversation_template=partial(
            ChatMLConversationTemplate,
            model="mistral",
            role_prefix=_chatml_role_prefix,
            prompt_format="<|{role}|>\n{text}</s>",
            inference_condition="",
        ),
    ),
}
