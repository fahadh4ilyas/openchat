from typing import Optional, Callable, Iterable, List
from transformers.tokenization_utils import PreTrainedTokenizer

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

    weight: Optional[float] = None

class ChatMLMessage(BaseModel):
    message: str
    weight: float


class Conversation(BaseModel):
    items: List[Message]

    condition: str = ""
    system: str = ""

class PretokenizedConversation(BaseModel):
    input_ids: List[int]
    loss_weights: List[float]


class ConversationTemplate(BaseModel):
    tokenizer: PreTrainedTokenizer

    # Prompt
    role_prefix: Callable[..., str]
    eot: str

    inference_condition: Optional[str] = None

    # Private
    bos_tokens_: List[int]
    eot_tokens_: List[int]
    eos_tokens_: List[int]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        tokenizer = data["tokenizer"]
        eot = data["eot"]
        bos_tokens_ = tokenizer("").input_ids
        eot_tokens_ = tokenizer(eot, add_special_tokens=False).input_ids
        eos_tokens_ = [tokenizer.eos_token_id]

        super().__init__(**data, bos_tokens_=bos_tokens_, eot_tokens_=eot_tokens_, eos_tokens_=eos_tokens_)

    def _safe_tokenize(self, strings: Iterable[str]) -> List[List[int]]:
        return self.tokenizer(strings, split_special_tokens=False, return_attention_mask=False, add_special_tokens=False).input_ids

    def tokenize_conversations(self, conversations: Iterable[Conversation], inference: bool = False, seq_level_weight: bool = False, force_eos_token: bool = False):
        # Pre-tokenize all conversations
        default_condition = self.inference_condition if inference else ""

        sys_mappings = set()
        role_mappings = set()
        all_text = []
        for conv in conversations:
            sys_mappings.add(conv.system)
            for msg in conv.items:
                role = msg.role
                if msg.name is not None:
                    role = msg.name
                role_mappings.add((role, conv.condition or default_condition))
                all_text.append(msg.content)

        sys_mappings = list(sys_mappings)
        role_mappings = list(role_mappings)

        # Tokenize
        sys_mappings = dict(zip(sys_mappings, self._safe_tokenize(sys_mappings)))
        role_mappings = dict(zip(role_mappings, self._safe_tokenize([self.role_prefix(*args) for args in role_mappings])))
        all_text = self._safe_tokenize(all_text)

        # Convert
        result_tokens = []
        result_weights = []
        all_text_idx = 0
        for conv in conversations:
            tokens = []
            weights = []

            # bos tokens
            if self.tokenizer.add_bos_token:
                tokens.extend(self.bos_tokens_)
                weights.extend([0.] * len(self.bos_tokens_))

            # System
            if conv.system:
                system = sys_mappings[conv.system]
                tokens.extend(system)
                weights.extend([0.] * len(system))

                tokens.extend(self.eot_tokens_)
                weights.extend([0.] * len(self.eot_tokens_))

            # Messages
            last_idx = len(conv.items) - 1
            for idx, msg in enumerate(conv.items):
                # Prefix
                role = role_mappings[(msg.role, conv.condition or default_condition)]
                tokens.extend(role)
                weights.extend([0.] * len(role))

                # Message
                text = all_text[all_text_idx]
                all_text_idx += 1

                # weight
                w = None
                if not inference:
                    assert msg.weight is not None

                    w = msg.weight
                    if seq_level_weight:
                        w /= len(text) + len(self.eot_tokens_) + (len(self.eos_tokens_) if self.eos_tokens_[0] != self.eot_tokens_[0] and force_eos_token else 0)

                # Message tokens
                tokens.extend(text)
                weights.extend([w] * len(text))

                if not (inference and idx == last_idx):  # Do not add EOT on last turn during inference
                    tokens.extend(self.eot_tokens_)
                    weights.extend([w] * len(self.eot_tokens_))
                    if self.eos_tokens_[0] != self.eot_tokens_[0] and force_eos_token:
                        tokens.extend(self.eos_tokens_)
                        weights.extend([w] * len(self.eos_tokens_))

            # Append result
            result_tokens.append(tokens)
            result_weights.append(weights)

        # Sanity check
        assert all_text_idx == len(all_text)

        return result_tokens, result_weights

class ChatMLConversationTemplate(BaseModel):
    tokenizer: PreTrainedTokenizer

    model: str

    # Prompt
    role_prefix: Callable[..., str]

    prompt_format: str
    conv_sep: List[int]
    sep: List[int]

    inference_condition: Optional[str] = None

    bos_tokens_: List[int]
    eos_tokens_: List[int]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        tokenizer = data["tokenizer"]

        conv_sep = tokenizer(data.pop('conv_sep', '\n'), add_special_tokens=False).input_ids
        sep = tokenizer(data.pop('sep', '\n'), add_special_tokens=False).input_ids
        bos_tokens_ = tokenizer("").input_ids
        eos_tokens_ = [tokenizer.eos_token_id]

        super().__init__(**data, conv_sep=conv_sep, sep=sep, bos_tokens_=bos_tokens_, eos_tokens_=eos_tokens_)
    
    def _safe_tokenize(self, strings: Iterable[str]) -> List[List[int]]:
        return self.tokenizer(strings, return_attention_mask=False, add_special_tokens=False).input_ids

    def _convert_to_chatml(self, conversation: Conversation, default_condition: str = "") -> List[ChatMLMessage]:

        prompts = []
        if conversation.system:
            prompts.append(ChatMLMessage(message=self.prompt_format.format(role='system', text=conversation.system), weight=0.0))
        
        for message in conversation.items:
            role = message.role
            if message.name is not None:
                role = message.name
            prompts.append(ChatMLMessage(message=self.prompt_format.format(role=self.role_prefix(role , conversation.condition or default_condition, self.model), text=message.content.strip()), weight=message.weight))
        
        return prompts

    def tokenize_conversations(self, conversations: Iterable[Conversation], inference: bool = False, seq_level_weight: bool = False, force_eos_token: bool = False, eos_final: bool = False):

        default_condition = self.inference_condition if inference else ""

        chatml_conversations = [self._convert_to_chatml(conv, default_condition) for conv in conversations]
        all_text = [msg.message for conv in chatml_conversations for msg in conv]
        text_mapping = dict(zip(all_text, self._safe_tokenize(all_text)))

        result_tokens = []
        result_weights = []
        for conv in chatml_conversations:
            tokens = []
            weights = []

            tokens.extend(self.bos_tokens_)
            weights.extend([0.0] * len(self.bos_tokens_))

            for msg in conv[:-1]:
                token_msg = text_mapping[msg.message]
                tokens.extend(token_msg)
                if msg.weight == 0:
                    weights.extend([0.0] * len(token_msg))
                else:
                    if force_eos_token and token_msg[-1] != self.eos_tokens_[0]:
                        tokens.extend(self.eos_tokens_)
                        token_msg.extend(self.eos_tokens_)
                    first_index = token_msg.index(self.sep[-1])
                    weights.extend([0.0] * first_index)
                    rest_index = len(token_msg[first_index:])
                    w = msg.weight
                    if seq_level_weight:
                        w /= rest_index
                    weights.extend([w] * rest_index)
                tokens.extend(self.conv_sep)
                weights.extend([0.0] * len(self.conv_sep))
            
            if len(conv) > 0:
                msg = conv[-1]
                token_msg = text_mapping[msg.message]
                tokens.extend(token_msg)
                if msg.weight == 0:
                    weights.extend([0.0] * len(token_msg))
                else:
                    if (force_eos_token or eos_final) and token_msg[-1] != self.eos_tokens_[0]:
                        tokens.extend(self.eos_tokens_)
                        token_msg.extend(self.eos_tokens_)
                    first_index = token_msg.index(self.sep[-1])
                    weights.extend([0.0] * first_index)
                    rest_index = len(token_msg[first_index:])
                    w = msg.weight
                    if seq_level_weight:
                        w /= rest_index
                    weights.extend([w] * rest_index)

            result_tokens.append(tokens)
            result_weights.append(weights)
    
        return result_tokens, result_weights
