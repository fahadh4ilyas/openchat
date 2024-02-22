from functools import partial

import torch
import transformers

from ochat.config.model_config import ModelConfig
from ochat.config.conversation_template import Message, Conversation, ConversationTemplate, ChatMLConversationTemplate
import ochat.models


_V3_2_PREFIXES = {
    # OpenAI mapping

    "user": "User:",
    "assistant": "Assistant:"
}


def _v3_2_role_prefix(from_role, condition):
    return f"{condition} {_V3_2_PREFIXES.get(from_role, from_role+':')}".strip()

def _chatml_role_prefix(from_role, condition):
    return f"{condition} {from_role}".strip()


MODEL_CONFIG_MAP = {
    # OpenChat V3.2
    "llama": ModelConfig(
        # Model
        model_max_context=4096,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),
        model_create_for_training=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4")
    ),

    "llamaYarn": ModelConfig(
        # Model
        model_max_context=16*4096,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),
        model_create_for_training=partial(ochat.models.LlamaYarnForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4")
    ),

    "mistral": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.MistralForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4 Correct")
    ),

    "mistralYarn": ModelConfig(
        # Model
        model_max_context=16*4096,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.MistralYarnForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4 Correct")
    ),

    "mixtral": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.MixtralForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4 Correct")
    ),

    "phi": ModelConfig(
        # Model
        model_max_context=2048,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.PhiForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4 Correct")
    ),

    "gemma": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.GemmaForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4 Correct")
    ),

    "llama_chatml": ModelConfig(
        # Model
        model_max_context=4096,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),
        model_create_for_training=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ChatMLConversationTemplate,
                                      role_prefix=_chatml_role_prefix,
                                      prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
                                      inference_condition="GPT4")
    ),

    "llamaYarn_chatml": ModelConfig(
        # Model
        model_max_context=16*4096,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),
        model_create_for_training=partial(ochat.models.LlamaYarnForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ChatMLConversationTemplate,
                                      role_prefix=_chatml_role_prefix,
                                      prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
                                      inference_condition="GPT4")
    ),

    "mistral_chatml": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.MistralForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ChatMLConversationTemplate,
                                      role_prefix=_chatml_role_prefix,
                                      prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
                                      inference_condition="GPT4 correct")
    ),

    "mistralYarn_chatml": ModelConfig(
        # Model
        model_max_context=16*4096,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.MistralYarnForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ChatMLConversationTemplate,
                                      role_prefix=_chatml_role_prefix,
                                      prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
                                      inference_condition="GPT4 correct")
    ),

    "mixtral_chatml": ModelConfig(
        # Model
        model_max_context=32768,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.MixtralForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ChatMLConversationTemplate,
                                      role_prefix=_chatml_role_prefix,
                                      prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
                                      inference_condition="GPT4 correct")
    ),

    "phi_chatml": ModelConfig(
        # Model
        model_max_context=2048,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.PhiForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ChatMLConversationTemplate,
                                      role_prefix=_chatml_role_prefix,
                                      prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
                                      inference_condition="GPT4 correct")
    ),

    "gemma_chatml": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.GemmaForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ChatMLConversationTemplate,
                                      role_prefix=_chatml_role_prefix,
                                      prompt_format="<|im_start|>{role}\n{text}<|im_end|>",
                                      inference_condition="GPT4 correct")
    ),

    "zephyr": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.MistralForCausalLM.from_pretrained,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ChatMLConversationTemplate,
                                      role_prefix=_chatml_role_prefix,
                                      prompt_format="<|{role}|>\n{text}</s>",
                                      inference_condition="")
    ),

}
