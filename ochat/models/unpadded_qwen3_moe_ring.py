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
""" PyTorch Unpadded & Fused Qwen3 Moe model. Compatible with HF. """

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    from ring_flash_attn import (
        zigzag_ring_flash_attn_func,
        zigzag_ring_flash_attn_varlen_func,
    )
except ImportError:
    print("FlashAttention not found. Install it if you need to train models.")

from ochat.kernel.rms_layernorm import fast_rms_layernorm
from ochat.kernel.rope import fast_rope_embedding


logger = logging.get_logger(__name__)


def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attn_length: int = None, max_length: int = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attn_length is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        attention_mask = torch.cat([
            torch.ones(attn_length, dtype=torch.long),
            torch.zeros(max_length - attn_length, dtype=torch.long) if max_length is not None else torch.tensor([], dtype=torch.long)
        ], dim=0)

        sequence_length = attention_mask.shape[0]
        num_hidden_layers = concatenated_gate_logits.shape[0] // sequence_length

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, None, None]
            .expand((num_hidden_layers, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, None]
            .expand((num_hidden_layers, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


@torch.jit.script  # type: ignore
def weighted_token_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
):
    return (weights * (torch.argmax(logits, dim=-1) == labels)).sum()


# @torch.jit.script  # type: ignore
def weighted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
):
    return (
        weights * cross_entropy_loss(logits, labels, inplace_backward=True)[0]
    ).sum()


@torch.jit.script  # type: ignore
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
    if fast_rope:
        return fast_rope_embedding(q, k, cos[position_ids], sin[position_ids])
    cos = cos[position_ids].unsqueeze(-2)  # [nnz, 1, head_dim]
    sin = sin[position_ids].unsqueeze(-2)  # [nnz, 1, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen3Moe
class UnpaddedQwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        """
        UnpaddedQwen3MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, use_fast_norm: bool = False):
        if use_fast_norm:
            return fast_rms_layernorm(hidden_states, self.weight, self.variance_epsilon)
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen3Moe
class UnpaddedQwen3MoeRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

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

    def forward(self, max_position_embeddings):
        if max_position_embeddings > self.max_position_embeddings:
            max_position_embeddings = -(-max_position_embeddings // 2048) * 2048
            self.calculate_cos_sin(max_position_embeddings)
        return self.cos_cached, self.sin_cached


class UnpaddedQwen3MoeMLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, intermediate_size=None):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)), None


class UnpaddedQwen3MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        num_experts = top_k_weights.shape[1]
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            _, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, expert_idx, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class UnpaddedQwen3MoeTopKRouter(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices


class UnpaddedQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = UnpaddedQwen3MoeExperts(config)
        self.router = UnpaddedQwen3MoeTopKRouter(config)

    def forward(self, nz_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routing_weights, selected_experts = self.router(nz_hidden_states)
        final_hidden_states = self.experts(nz_hidden_states, selected_experts, routing_weights)
        return final_hidden_states, routing_weights


class UnpaddedQwen3MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: Optional[int] = None):
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
        self.q_norm = UnpaddedQwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = UnpaddedQwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

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

        # flash attn
        if cu_seqlens[-1] == max_seqlen:
            attn_output = zigzag_ring_flash_attn_func(
                q=query_states.unsqueeze(0),
                k=key_states.unsqueeze(0),
                v=value_states.unsqueeze(0),
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
            )
        else:
            attn_output = zigzag_ring_flash_attn_varlen_func(
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
            )

        # attn_output: [total_nnz, num_heads, head_dim]
        attn_output = attn_output.view(-1, self.num_attention_heads * self.head_dim)  # type: ignore
        return self.o_proj(attn_output)


class UnpaddedQwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = UnpaddedQwen3MoeAttention(config=config, layer_idx=layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = UnpaddedQwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = UnpaddedQwen3MoeMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = UnpaddedQwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = UnpaddedQwen3MoeRMSNorm(
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
        nz_hidden_states, router_logits = self.mlp(nz_hidden_states)
        nz_hidden_states = residual + nz_hidden_states

        return nz_hidden_states, router_logits


class UnpaddedQwen3MoePreTrainedModel(PreTrainedModel):
    config_class = Qwen3MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnpaddedQwen3MoeDecoderLayer"]

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


class UnpaddedQwen3MoeModel(UnpaddedQwen3MoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UnpaddedQwen3MoeDecoderLayer`]

    Args:
        config: Qwen3MoeConfig
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.rotary_emb = UnpaddedQwen3MoeRotaryEmbedding(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.layers = nn.ModuleList(
            [
                UnpaddedQwen3MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = UnpaddedQwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        all_router_logits = ()

        # decoder layers
        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                nz_hidden_states, router_logits = self._gradient_checkpointing_func(
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
                nz_hidden_states, router_logits = decoder_layer(
                    cos_sin=cos_sin,
                    nz_hidden_states=nz_hidden_states,
                    nz_position_ids=nz_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    use_fast_norm=use_fast_norm,
                    use_fast_rope=use_fast_rope,
                )
            if router_logits is not None:
                all_router_logits += (router_logits,)

        nz_hidden_states = self.norm(nz_hidden_states, use_fast_norm)

        return nz_hidden_states, all_router_logits


class Qwen3MoeForCausalLM(UnpaddedQwen3MoePreTrainedModel):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.model = UnpaddedQwen3MoeModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

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
        total_seqs: float,
        # Unpadded labels
        nz_shifted_label_ids: Optional[torch.Tensor] = None,
        nz_shifted_loss_weights: Optional[torch.Tensor] = None,
        use_fast_norm: bool = False,
        use_fast_rope: bool = False,
    ) -> CausalLMOutputWithPast:
        # Model logits
        hidden_states, router_logits = self.model(
            nz_input_ids=nz_input_ids,
            nz_position_ids=nz_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            use_fast_norm=use_fast_norm,
            use_fast_rope=use_fast_rope,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if nz_shifted_label_ids is not None:
            assert nz_shifted_loss_weights is not None
            max_length = nz_input_ids.shape[0]
            latest_seq = nz_input_ids[cu_seqlens[-2]:cu_seqlens[-1]]
            if (latest_seq == 0).all():
                attn_length = cu_seqlens[-2]
            else:
                attn_length = cu_seqlens[-1]

            aux_loss = load_balancing_loss_func(
                router_logits, self.num_experts, self.num_experts_per_tok, attn_length=attn_length, max_length=max_length
            )

            acc = (
                weighted_token_accuracy(
                    logits.detach(), nz_shifted_label_ids, nz_shifted_loss_weights
                )
                / total_seqs
            )
            loss = (
                weighted_cross_entropy(
                    logits, nz_shifted_label_ids, nz_shifted_loss_weights
                )
                / total_seqs
                + self.router_aux_loss_coef * aux_loss,
                self.router_aux_loss_coef * aux_loss,
            )

        return CausalLMOutputWithPast(
            loss=(loss, acc),  # type: ignore
            logits=logits,
        )
