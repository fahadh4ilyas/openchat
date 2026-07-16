import json, json_repair, re
from typing import Optional, Callable, Iterable, List, Dict, Union
from transformers import PreTrainedTokenizerBase

from pydantic import BaseModel, field_validator, model_validator, ValidationInfo


class TextContentPart(BaseModel):
    type: str = "text"
    text: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v != "text":
            raise ValueError("Invalid type for TextContentPart")
        return v


class Url(BaseModel):
    url: str


class ImageContentPart(BaseModel):
    type: str
    image_url: Url

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v not in ["image", "image_url"]:
            raise ValueError("Invalid type for ImageContentPart")
        return v


class VideoContentPart(BaseModel):
    type: str
    video: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v != "video":
            raise ValueError("Invalid type for VideoContentPart")
        return v


class FunctionCall(BaseModel):
    name: str
    arguments: Union[Dict, str]

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, v, info: ValidationInfo):
        if isinstance(v, str):
            use_repair = info.context.get("use_json_repair", False)
            try:
                if use_repair:
                    v = json_repair.loads(v)
                else:
                    v = json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Arguments string is not a valid JSON")
            except Exception:
                raise ValueError("Error parsing arguments string as JSON")
        elif not isinstance(v, dict):
            raise ValueError("Arguments must be a dict or a JSON string")
        return v


class ToolCall(BaseModel):
    type: str
    function: FunctionCall

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v != "function":
            raise ValueError("Invalid type for ToolCall")
        return v


class MessageOpenAI(BaseModel):
    role: str
    reasoning_content: Optional[str] = None
    content: Union[None, str, List[TextContentPart | ImageContentPart | VideoContentPart]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    weight: Optional[float] = None

    @model_validator(mode="after")
    def validate_message(self):
        if isinstance(self.content, str):
            if self.role == "assistant":
                match_reasoning = re.search(r"<think>(.*?)</think>", self.content, re.DOTALL)
                if match_reasoning:
                    self.reasoning_content = match_reasoning.group(1).strip()
                    self.content = self.content.replace(match_reasoning.group(0), "").strip()
            self.content = [TextContentPart(text=self.content)]
        elif isinstance(self.content, list) and self.role == "assistant" and isinstance(self.content[0], TextContentPart):
                match_reasoning = re.search(r"<tool_call>(.*?)</tool_call>", self.content[0].text, re.DOTALL)
                if match_reasoning:
                    self.reasoning_content = match_reasoning.group(1).strip()
                    self.content[0].text = self.content[0].text.replace(match_reasoning.group(0), "").strip()
        elif self.content is None:
            self.content = []
        return self


class Function(BaseModel):
    name: str
    parameters: Optional[Dict] = None


class Tool(BaseModel):
    type: str
    function: Function

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v != "function":
            raise ValueError("Invalid type for Tool")
        return v


class ConversationOpenAI(BaseModel):
    messages: List[MessageOpenAI]
    tools: Optional[List[Tool]] = None


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

    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None

    condition: str = ""
    system: str = ""


class PretokenizedConversation(BaseModel):
    input_ids: List[int]
    loss_weights: List[float]

    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None


class PretrainingText(BaseModel):
    text: str
    weight: float = 1.0

    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None


class ConversationTemplate(BaseModel):
    tokenizer: PreTrainedTokenizerBase

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

        super().__init__(
            **data,
            bos_tokens_=bos_tokens_,
            eot_tokens_=eot_tokens_,
            eos_tokens_=eos_tokens_,
        )

    def _safe_tokenize(self, strings: Iterable[str]) -> List[List[int]]:
        return self.tokenizer(
            strings,
            split_special_tokens=False,
            return_attention_mask=False,
            add_special_tokens=False,
        ).input_ids

    def tokenize_conversations(
        self,
        conversations: Iterable[Conversation],
        inference: bool = False,
        seq_level_weight: bool = False,
        force_eos_token: bool = False,
        eos_final: bool = False,
        **kwargs,
    ):
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
        role_mappings = dict(
            zip(
                role_mappings,
                self._safe_tokenize(
                    [self.role_prefix(*args) for args in role_mappings]
                ),
            )
        )
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
                weights.extend([0.0] * len(self.bos_tokens_))

            # System
            if conv.system:
                system = sys_mappings[conv.system]
                tokens.extend(system)
                weights.extend([0.0] * len(system))

                tokens.extend(self.eot_tokens_)
                weights.extend([0.0] * len(self.eot_tokens_))

            # Messages
            last_idx = len(conv.items) - 1
            for idx, msg in enumerate(conv.items):
                # Prefix
                role = role_mappings[(msg.role, conv.condition or default_condition)]
                tokens.extend(role)
                weights.extend([0.0] * len(role))

                # Message
                text = all_text[all_text_idx]
                all_text_idx += 1

                # weight
                w = None
                if not inference:
                    assert msg.weight is not None

                    w = msg.weight
                    if seq_level_weight:
                        w /= (
                            len(text)
                            + len(self.eot_tokens_)
                            + (
                                len(self.eos_tokens_)
                                if self.eos_tokens_[0] != self.eot_tokens_[0]
                                and force_eos_token
                                else 0
                            )
                        )

                # Message tokens
                tokens.extend(text)
                weights.extend([w] * len(text))

                if not (
                    inference and idx == last_idx
                ):  # Do not add EOT on last turn during inference
                    tokens.extend(self.eot_tokens_)
                    weights.extend([w] * len(self.eot_tokens_))
                    if self.eos_tokens_[0] != self.eot_tokens_[0] and (
                        force_eos_token or eos_final
                    ):
                        tokens.extend(self.eos_tokens_)
                        weights.extend([w] * len(self.eos_tokens_))

            # Append result
            result_tokens.append(tokens)
            result_weights.append(weights)

        # Sanity check
        assert all_text_idx == len(all_text)

        return result_tokens, result_weights


class ChatMLConversationTemplate(BaseModel):
    tokenizer: PreTrainedTokenizerBase

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

        conv_sep = tokenizer(
            data.pop("conv_sep", "\n"), add_special_tokens=False
        ).input_ids
        sep = tokenizer(data.pop("sep", "\n"), add_special_tokens=False).input_ids
        bos_tokens_ = tokenizer("").input_ids
        eos_tokens_ = [tokenizer.eos_token_id]

        super().__init__(
            **data,
            conv_sep=conv_sep,
            sep=sep,
            bos_tokens_=bos_tokens_,
            eos_tokens_=eos_tokens_,
        )

    def _safe_tokenize(self, strings: Iterable[str]) -> List[List[int]]:
        return self.tokenizer(
            strings, return_attention_mask=False, add_special_tokens=False
        ).input_ids

    def _build_prompt_items(
        self,
        items: list,
        condition: str,
        default_condition: str,
        system: str = "",
        separate_think: bool = False,
    ) -> List[ChatMLMessage]:
        prompts = []

        if system:
            prompts.append(
                ChatMLMessage(
                    message=self.prompt_format.format(role="system", text=system),
                    weight=0.0,
                )
            )

        for message in items:
            role = message.role
            if message.name is not None:
                role = message.name
            weight_val = message.weight

            content = message.content
            if separate_think and role == "assistant" and weight_val == 0 and "<think>" in content:
                content = content.split("</think>")[-1].lstrip()

            prompts.append(
                ChatMLMessage(
                    message=self.prompt_format.format(
                        role=self.role_prefix(
                            role,
                            condition or default_condition,
                            self.model,
                        ),
                        text=content.strip(),
                    ),
                    weight=weight_val,
                )
            )

        return prompts

    def _convert_to_chatml(
        self, conversation: Conversation, default_condition: str = "", separate_think: bool = False
    ) -> List[List[ChatMLMessage]]:
        list_prompts = []

        if separate_think:
            weighted_indices = [i for i, msg in enumerate(conversation.items) if msg.weight is not None and msg.weight != 0 and '<think>' in msg.content]

            for idx in weighted_indices:
                temp_items = [item.model_copy(update={'weight': 0.0}, deep=True) for item in conversation.items[:idx]] + [conversation.items[idx].model_copy(deep=True)]
                prompts = self._build_prompt_items(
                    temp_items, conversation.condition, default_condition,
                    system=conversation.system, separate_think=True,
                )
                list_prompts.append(prompts)
        else:
            prompts = self._build_prompt_items(
                conversation.items, conversation.condition, default_condition,
                system=conversation.system,
            )
            list_prompts.append(prompts)

        return list_prompts

    def tokenize_conversations(
        self,
        conversations: Iterable[Conversation],
        inference: bool = False,
        seq_level_weight: bool = False,
        force_eos_token: bool = False,
        eos_final: bool = False,
        separate_think: bool = False,
    ):
        default_condition = self.inference_condition if inference else ""

        chatml_conversations = [
            c for conv in conversations for c in self._convert_to_chatml(conv, default_condition, separate_think)
        ]
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
                    if (force_eos_token or eos_final) and token_msg[
                        -1
                    ] != self.eos_tokens_[0]:
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

class DeepseekConversationTemplate(BaseModel):
    tokenizer: PreTrainedTokenizerBase

    prompt_format: Dict[str, str]

    bos_tokens_: List[int]
    eos_tokens_: List[int]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        tokenizer = data["tokenizer"]

        bos_tokens_ = tokenizer("").input_ids
        eos_tokens_ = [tokenizer.eos_token_id]

        super().__init__(
            **data,
            bos_tokens_=bos_tokens_,
            eos_tokens_=eos_tokens_,
        )
    
    def _safe_tokenize(self, strings: Iterable[str]) -> List[List[int]]:
        return self.tokenizer(
            strings, return_attention_mask=False, add_special_tokens=False
        ).input_ids

    def _convert_to_chatml(
        self, conversation: Conversation
    ) -> List[List[ChatMLMessage]]:
        list_prompts = []

        weighted_indices = [i for i, msg in enumerate(conversation.items) if msg.weight is not None and msg.weight != 0 and '<think>' in msg.content]

        for idx in weighted_indices:
            temp_conversation_items = [item.model_copy(update={'weight': 0.0}, deep=True) for item in conversation.items[:idx]] + [conversation.items[idx].model_copy(deep=True)]
            temp_conversation = Conversation(items=temp_conversation_items, condition=conversation.condition, system=conversation.system)

            prompts = []
            if temp_conversation.system:
                prompts.append(
                    ChatMLMessage(
                        message=conversation.system,
                        weight=0.0,
                    )
                )
            
            for message in temp_conversation.items:
                role = message.role
                if role not in ['user', 'assistant']:
                    raise ValueError(f"Role {role} is not supported")
                weight = message.weight
                if role == 'assistant' and weight == 0 and '<think>' in message.content:
                    content = message.content.split('</think>')[-1].lstrip()
                else:
                    content = message.content
                prompts.append(
                    ChatMLMessage(
                        message=self.prompt_format[role].format(
                            role=role.title(),
                            text=content.strip(),
                        ),
                        weight=weight,
                    )
                )
            list_prompts.append(prompts)
        
        return list_prompts

    def tokenize_conversations(
        self,
        conversations: Iterable[Conversation],
        inference: bool = False,
        seq_level_weight: bool = False,
        force_eos_token: bool = False,
        eos_final: bool = False,
        **kwargs,
    ):

        chatml_conversations = [
            c for conv in conversations for c in self._convert_to_chatml(conv)
        ]

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
                    first_index = 1
                    weights.extend([0.0] * first_index)
                    rest_index = len(token_msg[first_index:])
                    w = msg.weight
                    if seq_level_weight:
                        w /= rest_index
                    weights.extend([w] * rest_index)

            if len(conv) > 0:
                msg = conv[-1]
                token_msg = text_mapping[msg.message]
                tokens.extend(token_msg)
                if msg.weight == 0:
                    weights.extend([0.0] * len(token_msg))
                else:
                    if (force_eos_token or eos_final) and token_msg[
                        -1
                    ] != self.eos_tokens_[0]:
                        tokens.extend(self.eos_tokens_)
                        token_msg.extend(self.eos_tokens_)
                    first_index = 1
                    weights.extend([0.0] * first_index)
                    rest_index = len(token_msg[first_index:])
                    w = msg.weight
                    if seq_level_weight:
                        w /= rest_index
                    weights.extend([w] * rest_index)

            result_tokens.append(tokens)
            result_weights.append(weights)

        return result_tokens, result_weights