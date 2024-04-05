# coding=utf-8
# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" PyTorch Phi model."""


from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_phi import PhiConfig

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
except ImportError:
    print ("FlashAttention not found. Install it if you need to train models.")


logger = logging.get_logger(__name__)


@torch.jit.script  # type: ignore
def weighted_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    return (weights * (torch.argmax(logits, dim=-1) == labels)).sum()


@torch.jit.script  # type: ignore
def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    return (weights * torch.nn.functional.cross_entropy(logits, labels, reduction="none")).sum()


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor):
    # q, k:     [nnz, num_heads, head_dim]
    # position_ids: [nnz]
    # cos, sin: [max_seq_len, head_dim]
    cos = cos[position_ids].unsqueeze(-2)  # [nnz, 1, head_dim]
    sin = sin[position_ids].unsqueeze(-2)  # [nnz, 1, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Phi
class UnpaddedPhiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, dtype=torch.int64, device=device).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class UnpaddedPhiLinearScalingRotaryEmbedding(torch.nn.Module):
    """PhiRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.int64, device=device).type_as(inv_freq)
        t = t / scaling_factor
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Phi
class UnpaddedPhiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.activation_fn = ACT2FN[config.hidden_act]
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gate_proj(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states


class UnpaddedPhiAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PhiConfig):
        super().__init__()

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.rotary_dim = int(config.partial_rotary_factor * self.head_dim)
        self.num_key_value_heads = config.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
            self.k_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )

    # Phi-2 has an attention overflow issue (with FP16) and requires autocast to be disabled
    @torch.autocast("cpu", enabled=False)
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        nz_position_ids: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int
    ) -> torch.Tensor:

        query_states = self.q_proj(nz_hidden_states).view(-1, self.num_heads, self.head_dim)
        key_states = self.k_proj(nz_hidden_states).view(-1,   self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(nz_hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_dim],
            query_states[..., self.rotary_dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_dim],
            key_states[..., self.rotary_dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, nz_position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        attn_output = flash_attn_varlen_func(
            q=query_states, k=key_states, v=value_states,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,

            dropout_p=self.attention_dropout, causal=True)

        attn_output = attn_output.view(-1, self.hidden_size)  # type: ignore
        return self.o_proj(attn_output)


class UnpaddedPhiDecoderLayer(nn.Module):
    def __init__(self, config: PhiConfig):
        super().__init__()
        self.self_attn = UnpaddedPhiAttention(config)
        self.mlp = UnpaddedPhiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        nz_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int
    ) -> torch.Tensor:

        residual = nz_hidden_states

        nz_hidden_states = self.input_layernorm(nz_hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            cos_sin=cos_sin,

            nz_hidden_states=nz_hidden_states,
            nz_position_ids=nz_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(nz_hidden_states))
        nz_hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return nz_hidden_states


class UnpaddedPhiPreTrainedModel(PreTrainedModel):
    config_class = PhiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnpaddedPhiDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class UnpaddedPhiModel(UnpaddedPhiPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhiDecoderLayer`]

    Args:
        config: PhiConfig
    """

    def __init__(self, config: PhiConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.partial_rotary_factor = config.partial_rotary_factor

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        if config.rope_scaling is None:
            self.rotary_emb   = UnpaddedPhiRotaryEmbedding(int(self.partial_rotary_factor * config.hidden_size // config.num_attention_heads),
                                                            max_position_embeddings=config.max_position_embeddings,
                                                            base=config.rope_theta)
        elif config.rope_scaling["type"] == "linear":
            self.rotary_emb = UnpaddedPhiLinearScalingRotaryEmbedding(int(self.partial_rotary_factor * config.hidden_size // config.num_attention_heads),
                                                                      max_position_embeddings=config.max_position_embeddings,
                                                                      base=config.rope_theta,
                                                                      scaling_factor=config.rope_scaling["factor"])
        self.layers = nn.ModuleList(
            [UnpaddedPhiDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        # Unpadded inputs
        nz_input_ids: torch.Tensor,
        nz_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        nz_hidden_states = self.embed_tokens(nz_input_ids)
        nz_hidden_states = self.embed_dropout(nz_hidden_states)
        cos_sin          = self.rotary_emb()

        # decoder layers
        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                nz_hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    cos_sin,

                    nz_hidden_states,
                    nz_position_ids,
                    cu_seqlens,
                    max_seqlen
                )
            else:
                nz_hidden_states = decoder_layer(
                    cos_sin=cos_sin,
                    
                    nz_hidden_states=nz_hidden_states,
                    nz_position_ids=nz_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen
                )

        nz_hidden_states = self.final_layernorm(nz_hidden_states)

        return nz_hidden_states


class PhiForCausalLM(UnpaddedPhiPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ with Llama->Phi,bias=False->bias=True
    def __init__(self, config):
        super().__init__(config)
        self.model = UnpaddedPhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
        return self.model

    def forward(
        self,
        # Unpadded inputs
        nz_input_ids: torch.Tensor,
        nz_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        # Unpadded labels
        nz_shifted_label_ids: Optional[torch.Tensor] = None,
        nz_shifted_loss_weights:      Optional[torch.Tensor] = None,
        num_seq: Optional[int] = 0
    ) -> CausalLMOutputWithPast:
        # Model logits
        hidden_states = self.model(
            nz_input_ids=nz_input_ids,
            nz_position_ids=nz_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if nz_shifted_label_ids is not None:
            assert nz_shifted_loss_weights is not None

            if num_seq > 0:
                loss = weighted_cross_entropy(logits, nz_shifted_label_ids, nz_shifted_loss_weights) / num_seq, \
                    weighted_token_accuracy(logits.detach(), nz_shifted_label_ids, nz_shifted_loss_weights) / num_seq
            else:
                loss = weighted_cross_entropy(logits, nz_shifted_label_ids, nz_shifted_loss_weights), \
                    weighted_token_accuracy(logits.detach(), nz_shifted_label_ids, nz_shifted_loss_weights)

        return CausalLMOutputWithPast(
            loss=loss,  # type: ignore
            logits=logits
        )


class PaddedPhiForCausalLM(PhiForCausalLM):
    """Compat layer for padded inputs"""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        # unused
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ):
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        # get indices
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = int(seqlens_in_batch.max().item())
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

        # Unpad inputs
        nz_input_ids    = torch.take_along_dim(input_ids,    indices)
        nz_position_ids = torch.take_along_dim(position_ids, indices)

        # Unpadded forward
        logits = super().forward(
            nz_input_ids=nz_input_ids,
            nz_position_ids=nz_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen_in_batch
        ).logits

        # Pad logits
        logits = pad_input(logits, indices, batch_size, seq_len)

        return CausalLMOutputWithPast(logits=logits)  # type: ignore

    def prepare_inputs_for_generation(self,
                                      input_ids: torch.Tensor,
                                      **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask"),
            "position_ids": kwargs.get("position_ids")
        }