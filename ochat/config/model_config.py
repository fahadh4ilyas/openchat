from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
from functools import partial
from typing import Callable, Union
from .conversation_template import ConversationTemplate, ChatMLConversationTemplate

from pydantic import BaseModel


class ModelConfig(BaseModel):
    # Model
    model_max_context: int
    model_tokenizer_create: Callable[..., PreTrainedTokenizerBase]
    model_create_for_training: Callable[..., PreTrainedModel]

    # conversation template
    conversation_template: Union[
        partial[ConversationTemplate], partial[ChatMLConversationTemplate]
    ]

    class Config:
        arbitrary_types_allowed = True
