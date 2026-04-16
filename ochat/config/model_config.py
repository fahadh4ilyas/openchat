from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from functools import partial
from typing import Callable, Union
from .conversation_template import ConversationTemplate, ChatMLConversationTemplate, DeepseekConversationTemplate

from pydantic import BaseModel


class ModelConfig(BaseModel):
    # Model
    model_max_context: int
    model_tokenizer_create: Callable[..., Union[PreTrainedTokenizerBase, ProcessorMixin]]
    model_create_for_training: Callable[..., PreTrainedModel]
    model_has_processor: bool = False

    # conversation template
    conversation_template: Union[
        partial[ConversationTemplate], partial[ChatMLConversationTemplate], partial[DeepseekConversationTemplate]
    ]

    class Config:
        arbitrary_types_allowed = True
