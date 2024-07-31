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
""" PyTorch Unpadded & Fused Mixtral model. Compatible with HF. """

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.mixtral.configuration_mixtral import MixtralConfig

try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    from flash_attn.bert_padding import pad_input
except ImportError:
    print("FlashAttention not found. Install it if you need to train models.")

from ochat.kernel.rms_layernorm import fast_rms_layernorm
from ochat.kernel.rope import fast_rope_embedding


logger = logging.get_logger(__name__)


def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2
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

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

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


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mixtral
class UnpaddedMixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        """
        UnpaddedMixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, use_fast_norm: bool = False):
        if use_fast_norm:
            return fast_rms_layernorm(self, hidden_states)
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mixtral
class UnpaddedMixtralRotaryEmbedding(torch.nn.Module):
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


class UnpaddedMixtralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MixtralConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        nz_position_ids: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        use_fast_rope: bool = False,
    ) -> torch.Tensor:
        # nz_hidden_states: [nnz, num_heads, head_dim]
        # nz_position_ids:  [nnz]
        # cu_seqlens:       [bs + 1]

        query_states = self.q_proj(nz_hidden_states).view(
            -1, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(nz_hidden_states).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(nz_hidden_states).view(
            -1, self.num_key_value_heads, self.head_dim
        )

        # RoPE
        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, nz_position_ids, use_fast_rope
        )

        # flash attn
        if cu_seqlens[-1] == max_seqlen:
            attn_output = flash_attn_func(
                q=query_states.unsqueeze(0),
                k=key_states.unsqueeze(0),
                v=value_states.unsqueeze(0),
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
                window_size=(self.sliding_window, self.sliding_window)
                if self.sliding_window is not None
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
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
                window_size=(self.sliding_window, self.sliding_window)
                if self.sliding_window is not None
                else (-1, -1),
            )

        # attn_output: [total_nnz, num_heads, head_dim]
        attn_output = attn_output.view(-1, self.hidden_size)  # type: ignore
        return self.o_proj(attn_output)


class UnpaddedMixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, nz_hidden_states):
        current_nz_hidden_states = self.act_fn(self.w1(nz_hidden_states)) * self.w3(
            nz_hidden_states
        )
        current_nz_hidden_states = self.w2(current_nz_hidden_states)
        return current_nz_hidden_states


class UnpaddedMixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [UnpaddedMixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, nz_hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        sequence_length, hidden_dim = nz_hidden_states.shape
        nz_hidden_states = nz_hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(nz_hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(nz_hidden_states.dtype)

        final_nz_hidden_states = torch.zeros(
            (sequence_length, hidden_dim),
            dtype=nz_hidden_states.dtype,
            device=nz_hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = nz_hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_nz_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x_list, idx_list, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_nz_hidden_states.index_add_(
                0, top_x, current_nz_hidden_states.to(nz_hidden_states.dtype)
            )
        final_nz_hidden_states = final_nz_hidden_states.reshape(
            sequence_length, hidden_dim
        )
        return final_nz_hidden_states, router_logits


class UnpaddedMixtralDecoderLayer(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = UnpaddedMixtralAttention(config=config)
        self.block_sparse_moe = UnpaddedMixtralSparseMoeBlock(config=config)
        self.input_layernorm = UnpaddedMixtralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = UnpaddedMixtralRMSNorm(
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
        )
        nz_hidden_states = residual + nz_hidden_states

        # Fully Connected
        residual = nz_hidden_states

        nz_hidden_states = self.post_attention_layernorm(
            nz_hidden_states, use_fast_norm
        )
        nz_hidden_states, router_logits = self.block_sparse_moe(nz_hidden_states)
        nz_hidden_states = residual + nz_hidden_states

        return nz_hidden_states, router_logits


class UnpaddedMixtralPreTrainedModel(PreTrainedModel):
    config_class = MixtralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnpaddedMixtralDecoderLayer"]

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


class UnpaddedMixtralModel(UnpaddedMixtralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UnpaddedMixtralDecoderLayer`]

    Args:
        config: MixtralConfig
    """

    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.rotary_emb = UnpaddedMixtralRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=2048,
            base=config.rope_theta,
        )

        self.layers = nn.ModuleList(
            [
                UnpaddedMixtralDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = UnpaddedMixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            all_router_logits += (router_logits,)

        nz_hidden_states = self.norm(nz_hidden_states, use_fast_norm)

        return nz_hidden_states, all_router_logits


class MixtralForCausalLM(UnpaddedMixtralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = UnpaddedMixtralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
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
        # Unpadded labels
        nz_shifted_label_ids: Optional[torch.Tensor] = None,
        nz_shifted_loss_weights: Optional[torch.Tensor] = None,
        num_seq: int = 0,
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

            aux_loss = load_balancing_loss_func(
                router_logits, self.num_experts, self.num_experts_per_tok
            )

            if num_seq > 0:
                acc = (
                    weighted_token_accuracy(
                        logits.detach(), nz_shifted_label_ids, nz_shifted_loss_weights
                    )
                    / num_seq
                )
                loss = (
                    (
                        weighted_cross_entropy(
                            logits, nz_shifted_label_ids, nz_shifted_loss_weights
                        )
                        / num_seq
                        + self.router_aux_loss_coef * aux_loss,
                        self.router_aux_loss_coef * aux_loss,
                    ),
                    acc,
                )
            else:
                acc = weighted_token_accuracy(
                    logits.detach(), nz_shifted_label_ids, nz_shifted_loss_weights
                )
                loss = (
                    (
                        weighted_cross_entropy(
                            logits, nz_shifted_label_ids, nz_shifted_loss_weights
                        )
                        + self.router_aux_loss_coef * aux_loss,
                        self.router_aux_loss_coef * aux_loss,
                    ),
                    acc,
                )

        return CausalLMOutputWithPast(
            loss=loss,  # type: ignore
            logits=logits,
        )
