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

    # Auto-LR: scaling factor relative to the llama 7B reference point.
    # 1.0 = use default base_lr as-is; < 1.0 = reduce LR for larger/more-sensitive models.
    base_lr_scale: float = 1.0

    # conversation template
    conversation_template: Union[
        partial[ConversationTemplate], partial[ChatMLConversationTemplate], partial[DeepseekConversationTemplate]
    ]

    class Config:
        arbitrary_types_allowed = True
