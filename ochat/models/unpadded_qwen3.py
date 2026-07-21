# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch Unpadded & Fused Qwen3 model. Compatible with HF. """

from typing import Optional, Tuple

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
except ImportError:
    print("FlashAttention not found. Install it if you need to train models.")

from ochat.kernel.rms_layernorm import fast_rms_layernorm
from ochat.kernel.rope import fast_rope_embedding


logger = logging.get_logger(__name__)


@torch.compile  # type: ignore
def weighted_token_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
):
    return (weights * (torch.argmax(logits, dim=-1) == labels)).sum()


from ochat.training_utils._ce_utils import weighted_cross_entropy


@torch.compile  # type: ignore
def rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    fast_rope: bool = False,
):
    # q, k:     [nnz, num_heads, head_dim]
    # position_ids: [nnz]
    # cos, sin: [max_seq_len, head_dim]
    base_dtype = q.dtype
    if fast_rope:
        q_embed, k_embed = fast_rope_embedding(q, k, cos[position_ids], sin[position_ids])
    else:
        cos = cos[position_ids].unsqueeze(-2)  # [nnz, 1, head_dim]
        sin = sin[position_ids].unsqueeze(-2)  # [nnz, 1, head_dim]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(base_dtype), k_embed.to(base_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen3
class UnpaddedQwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        """
        UnpaddedQwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, use_fast_norm: bool = False):
        if use_fast_norm:
            return fast_rms_layernorm(hidden_states, self.weight, self.variance_epsilon)
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen3
class UnpaddedQwen3RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        self.dim = dim
        self.base = base
        self._initial_max_position_embeddings = max_position_embeddings

        # RoPE
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.device = device
        self.calculate_cos_sin(max_position_embeddings)

    def calculate_cos_sin(self, max_position_embeddings):
        self.max_position_embeddings = max_position_embeddings

        t = torch.arange(
            max_position_embeddings, dtype=torch.int64, device=self.device
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def reset_parameters(self):
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=self.inv_freq.device).float() / self.dim)
        )
        self.inv_freq.copy_(inv_freq)
        self.calculate_cos_sin(self._initial_max_position_embeddings)

    def forward(self, max_position_embeddings):
        if max_position_embeddings > self.max_position_embeddings:
            max_position_embeddings = -(-max_position_embeddings // 2048) * 2048
            self.calculate_cos_sin(max_position_embeddings)
        return self.cos_cached, self.sin_cached


class UnpaddedQwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnpaddedQwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.max_window_layers = config.max_window_layers
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.use_sliding_window = config.use_sliding_window

        self.q_proj = nn.Linear(
            self.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self.q_norm = UnpaddedQwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = UnpaddedQwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        nz_position_ids: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        use_fast_rope: bool = False,
        use_fast_norm: bool = False,
    ) -> torch.Tensor:
        # nz_hidden_states: [nnz, num_heads, head_dim]
        # nz_position_ids:  [nnz]
        # cu_seqlens:       [bs + 1]

        input_shape = nz_hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(nz_hidden_states).view(
            *hidden_shape
        ), use_fast_norm)
        key_states = self.k_norm(self.k_proj(nz_hidden_states).view(
            *hidden_shape
        ), use_fast_norm)
        value_states = self.v_proj(nz_hidden_states).view(
            *hidden_shape
        )

        # RoPE
        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, nz_position_ids, use_fast_rope
        )

        use_sliding_window = (
            self.use_sliding_window and self.layer_idx < self.max_window_layers
        )

        # flash attn
        query_states = query_states.to(torch.bfloat16)
        key_states = key_states.to(torch.bfloat16)
        value_states = value_states.to(torch.bfloat16)
        if cu_seqlens[-1] == max_seqlen:
            attn_output = flash_attn_func(
                q=query_states.unsqueeze(0),
                k=key_states.unsqueeze(0),
                v=value_states.unsqueeze(0),
                softmax_scale=self.scaling,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
                window_size=(self.sliding_window, self.sliding_window)
                if use_sliding_window
                else (-1, -1),
            )
        else:
            attn_output = flash_attn_varlen_func(
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=self.scaling,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
                window_size=(self.sliding_window, self.sliding_window)
                if use_sliding_window
                else (-1, -1),
            )

        # attn_output: [total_nnz, num_heads, head_dim]
        attn_output = attn_output.view(-1, self.num_attention_heads * self.head_dim)  # type: ignore
        return self.o_proj(attn_output)


class UnpaddedQwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: Optional[int] = None):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = UnpaddedQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = UnpaddedQwen3MLP(config=config)
        self.input_layernorm = UnpaddedQwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = UnpaddedQwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        nz_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        use_fast_norm: bool = False,
        use_fast_rope: bool = False,
    ) -> torch.Tensor:
        # Self Attention
        residual = nz_hidden_states

        nz_hidden_states = self.input_layernorm(nz_hidden_states, use_fast_norm)
        nz_hidden_states = self.self_attn(
            cos_sin=cos_sin,
            nz_hidden_states=nz_hidden_states,
            nz_position_ids=nz_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            use_fast_rope=use_fast_rope,
            use_fast_norm=use_fast_norm,
        )
        nz_hidden_states = residual + nz_hidden_states

        # Fully Connected
        residual = nz_hidden_states

        nz_hidden_states = self.post_attention_layernorm(
            nz_hidden_states, use_fast_norm
        )
        nz_hidden_states = self.mlp(nz_hidden_states)
        nz_hidden_states = residual + nz_hidden_states

        return nz_hidden_states


class UnpaddedQwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnpaddedQwen3DecoderLayer"]

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

    def _initialize_missing_keys(self, is_quantized: bool) -> None:
        super()._initialize_missing_keys(is_quantized)
        for module in self.modules():
            if isinstance(module, UnpaddedQwen3RotaryEmbedding):
                module.reset_parameters()


class UnpaddedQwen3Model(UnpaddedQwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UnpaddedQwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        rope_theta = config.rope_theta if hasattr(config, "rope_theta") else config.rope_parameters["rope_theta"]
        self.rotary_emb = UnpaddedQwen3RotaryEmbedding(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            max_position_embeddings=2048,
            base=rope_theta,
        )

        self.layers = nn.ModuleList(
            [
                UnpaddedQwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = UnpaddedQwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        use_fast_norm: bool = False,
        use_fast_rope: bool = False,
    ) -> torch.Tensor:
        nz_hidden_states = self.embed_tokens(nz_input_ids)
        cos_sin = self.rotary_emb(max_seqlen)

        # decoder layers
        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                nz_hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    cos_sin,
                    nz_hidden_states,
                    nz_position_ids,
                    cu_seqlens,
                    max_seqlen,
                    use_fast_norm,
                    use_fast_rope,
                )
            else:
                nz_hidden_states = decoder_layer(
                    cos_sin=cos_sin,
                    nz_hidden_states=nz_hidden_states,
                    nz_position_ids=nz_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    use_fast_norm=use_fast_norm,
                    use_fast_rope=use_fast_rope,
                )

        nz_hidden_states = self.norm(nz_hidden_states, use_fast_norm)

        return nz_hidden_states


class Qwen3ForCausalLM(UnpaddedQwen3PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config):
        super().__init__(config)
        self.model = UnpaddedQwen3Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

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
        nz_shifted_loss_weights: Optional[torch.Tensor] = None,
        num_seq: int = 0,
        chunk_size: int = -1,
        use_fast_norm: bool = False,
        use_fast_rope: bool = False,
        return_per_seq_logps: bool = False,
    ) -> CausalLMOutputWithPast:
        # Model logits
        hidden_states = self.model(
            nz_input_ids=nz_input_ids,
            nz_position_ids=nz_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            use_fast_norm=use_fast_norm,
            use_fast_rope=use_fast_rope,
        )

        loss = None
        per_seq_logps = None
        if nz_shifted_label_ids is not None:
            assert nz_shifted_loss_weights is not None
            
            total_loss = 0.0
            total_acc = 0.0
            
            if return_per_seq_logps:
                _num_seq = int(cu_seqlens.shape[0] - 1)
                _all_token_losses: list = []
                _all_token_indices: list = []

            # Iterate through the sequence in chunks
            if chunk_size > 0:
                for i in range(0, hidden_states.size(0), chunk_size):
                    # 1. Grab chunks
                    hidden_chunk = hidden_states[i : i + chunk_size]
                    label_chunk = nz_shifted_label_ids[i : i + chunk_size]
                    weight_chunk = nz_shifted_loss_weights[i : i + chunk_size]
                    
                    # 2. Project ONLY this chunk to vocab size
                    logits_chunk = self.lm_head(hidden_chunk)
                    
                    # 3. Compute loss and accuracy for this chunk
                    if return_per_seq_logps:
                        token_losses = weighted_cross_entropy(
                            logits_chunk, label_chunk, weight_chunk, reduction="none"
                        )
                        chunk_loss = token_losses.sum()
                        positions = torch.arange(i, i + logits_chunk.size(0), device=logits_chunk.device)
                        seq_indices = torch.searchsorted(cu_seqlens, positions, right=True) - 1
                        _all_token_losses.append(token_losses)
                        _all_token_indices.append(seq_indices)
                    else:
                        chunk_loss = weighted_cross_entropy(logits_chunk, label_chunk, weight_chunk)
                    chunk_acc = weighted_token_accuracy(logits_chunk.detach(), label_chunk, weight_chunk)
                    
                    # 4. Accumulate
                    total_loss += chunk_loss
                    total_acc += chunk_acc
                    
                    # 5. Free the massive chunk from VRAM immediately
                    del logits_chunk
                    del hidden_chunk
            else:
                logits = self.lm_head(hidden_states)
                if return_per_seq_logps:
                    token_losses = weighted_cross_entropy(
                        logits, nz_shifted_label_ids, nz_shifted_loss_weights, reduction="none"
                    )
                    total_loss = token_losses.sum()
                    positions = torch.arange(logits.size(0), device=logits.device)
                    seq_indices = torch.searchsorted(cu_seqlens, positions, right=True) - 1
                    _all_token_losses.append(token_losses)
                    _all_token_indices.append(seq_indices)
                else:
                    total_loss = weighted_cross_entropy(logits, nz_shifted_label_ids, nz_shifted_loss_weights)
                total_acc = weighted_token_accuracy(logits.detach(), nz_shifted_label_ids, nz_shifted_loss_weights)

            if return_per_seq_logps:
                all_losses = torch.cat(_all_token_losses)
                all_indices = torch.cat(_all_token_indices)
                per_seq_loss = torch.zeros(_num_seq, device=all_losses.device, dtype=all_losses.dtype)
                per_seq_loss.index_add_(0, all_indices, all_losses)
                per_seq_logps = -per_seq_loss


            # Finalize metrics
            if num_seq > 0:
                loss = (total_loss / num_seq, total_acc / num_seq)
            else:
                loss = (total_loss, total_acc)

        return CausalLMOutputWithPast(
            loss=loss,  # type: ignore
            logits=per_seq_logps,
        )
