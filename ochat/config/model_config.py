from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Callable, Union, Type
from .conversation_template import ConversationTemplate, ChatMLConversationTemplate

from pydantic import BaseModel


class ModelConfig(BaseModel):

    # Model
    model_max_context: int
    model_tokenizer_create: Callable[..., PreTrainedTokenizer]
    model_create_for_training: Callable[..., PreTrainedModel]

    # conversation template
    conversation_template: Union[Type[ConversationTemplate], Type[ChatMLConversationTemplate]]

    class Config:
        arbitrary_types_allowed = True
