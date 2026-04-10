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
""" PyTorch Unpadded & Fused LLaMA model. Compatible with HF. """

from typing import Optional, Tuple

import torch, math
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_llama_long import LlamaConfig

try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
except ImportError:
    print("FlashAttention not found. Install it if you need to train models.")

from ochat.kernel.rms_layernorm import fast_rms_layernorm
from ochat.kernel.rope import fast_rope_embedding


logger = logging.get_logger(__name__)


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


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim, inv_freq):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.int64).type_as(inv_freq) - min) / (
        max - min
    )
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


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


class UnpaddedLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        """
        UnpaddedLlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, use_fast_norm: bool = False):
        if use_fast_norm:
            return fast_rms_layernorm(hidden_states, self.weight, self.variance_epsilon)
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class UnpaddedLlamaLinearScalingRotaryEmbedding(torch.nn.Module):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.device = device
        self.scaling_factor = scaling_factor
        self.calculate_cos_sin(max_position_embeddings)

    def calculate_cos_sin(self, max_position_embeddings):
        self.max_position_embeddings = max_position_embeddings

        t = torch.arange(
            max_position_embeddings, dtype=torch.int64, device=self.device
        ).type_as(self.inv_freq)
        t = t / self.scaling_factor
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


class UnpaddedLlamaYarnRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scale=1,
        original_max_position_embeddings=2048,
        extrapolation_factor=1,
        attn_factor=1,
        beta_fast=32,
        beta_slow=1,
        finetuned=False,
        device=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.yarn(device)

    def calculate_cos_sin(self, max_position_embeddings):
        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached, dtype=torch.int64, device=self.inv_freq.device
        ).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        self.register_buffer(
            "cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False
        )

    def yarn(self, device):
        pos_freqs = self.base ** (
            torch.arange(0, self.dim, 2, dtype=torch.int64, device=device).float()
            / self.dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = (
            (
                1
                - _yarn_linear_ramp_mask(
                    low, high, self.dim // 2, inv_freq_interpolation
                ).to(device)
            )
            * self.extrapolation_factor
        )  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mscale = float(
            _yarn_get_mscale(self.scale) * self.attn_factor
        )  # Get n-d magnitude scaling corrected for interpolation

    def forward(self, max_position_embeddings):
        if max_position_embeddings > self.max_seq_len_cached:
            max_position_embeddings = -(-max_position_embeddings // 2048) * 2048
            self.calculate_cos_sin(max_position_embeddings)
        return self.cos_cached, self.sin_cached


class UnpaddedLlama3RotaryEmbedding(torch.nn.Module):
    """LlamaRotaryEmbedding with Llama3 Scaling"""

    def __init__(
        self,
        dim,
        max_position_embeddings=131072,
        base=10000,
        device=None,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
        )
        low_freq_wavelen = original_max_position_embeddings / low_freq_factor
        high_freq_wavelen = original_max_position_embeddings / high_freq_factor
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (
                    original_max_position_embeddings / wavelen - low_freq_factor
                ) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
        inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)
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


class UnpaddedLlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnpaddedLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = (
            config.num_key_value_heads
            if config.num_key_value_heads is not None
            else config.num_attention_heads
        )
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
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
            )

        # attn_output: [total_nnz, num_heads, head_dim]
        attn_output = attn_output.view(-1, self.hidden_size)  # type: ignore
        return self.o_proj(attn_output)


class UnpaddedLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = UnpaddedLlamaAttention(config=config)
        self.mlp = UnpaddedLlamaMLP(config=config)
        self.input_layernorm = UnpaddedLlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = UnpaddedLlamaRMSNorm(
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
        nz_hidden_states = self.mlp(nz_hidden_states)
        nz_hidden_states = residual + nz_hidden_states

        return nz_hidden_states


class UnpaddedLlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnpaddedLlamaDecoderLayer"]

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


class UnpaddedLlamaModel(UnpaddedLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UnpaddedLlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        rope_scaling_type = config.rope_scaling.get(
            "type", None
        ) or config.rope_scaling.get("rope_type", None)
        if rope_scaling_type == "linear":
            self.rotary_emb = UnpaddedLlamaLinearScalingRotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                max_position_embeddings=2048,
                base=config.rope_theta,
                scaling_factor=config.rope_scaling["factor"],
            )
        elif rope_scaling_type == "yarn":
            self.rotary_emb = UnpaddedLlamaYarnRotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                max_position_embeddings=2048,
                scale=config.rope_scaling["factor"],
                original_max_position_embeddings=config.rope_scaling[
                    "original_max_position_embeddings"
                ],
                base=config.rope_theta,
            )
        elif rope_scaling_type == "llama3":
            self.rotary_emb = UnpaddedLlama3RotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                max_position_embeddings=2048,
                factor=config.rope_scaling["factor"],
                low_freq_factor=config.rope_scaling["low_freq_factor"],
                high_freq_factor=config.rope_scaling["high_freq_factor"],
                original_max_position_embeddings=config.rope_scaling[
                    "original_max_position_embeddings"
                ],
                base=config.rope_theta,
            )

        self.layers = nn.ModuleList(
            [UnpaddedLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = UnpaddedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class LlamaForCausalLM(UnpaddedLlamaPreTrainedModel):
    # Ignore rotary emb inv_freq on load, as they will be calculated on creation
    _keys_to_ignore_on_load_unexpected = [
        r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq"
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = UnpaddedLlamaModel(config)

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
        use_fast_norm: bool = False,
        use_fast_rope: bool = False,
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
        if nz_shifted_label_ids is not None:
            assert nz_shifted_loss_weights is not None
            
            chunk_size = 4096
            total_loss = 0.0
            total_acc = 0.0
            
            # Iterate through the sequence in chunks
            for i in range(0, hidden_states.size(0), chunk_size):
                # 1. Grab chunks
                hidden_chunk = hidden_states[i : i + chunk_size]
                label_chunk = nz_shifted_label_ids[i : i + chunk_size]
                weight_chunk = nz_shifted_loss_weights[i : i + chunk_size]
                
                # 2. Project ONLY this chunk to vocab size
                logits_chunk = self.lm_head(hidden_chunk)
                
                # 3. Compute loss and accuracy for this chunk
                chunk_loss = weighted_cross_entropy(logits_chunk, label_chunk, weight_chunk)
                chunk_acc = weighted_token_accuracy(logits_chunk.detach(), label_chunk, weight_chunk)
                
                # 4. Accumulate
                total_loss += chunk_loss
                total_acc += chunk_acc
                
                # 5. Free the massive chunk from VRAM immediately
                del logits_chunk
                del hidden_chunk

            # Finalize metrics
            if num_seq > 0:
                loss = (total_loss / num_seq, total_acc / num_seq)
            else:
                loss = (total_loss, total_acc)

        return CausalLMOutputWithPast(
            loss=loss,  # type: ignore
            logits=None,
        )
