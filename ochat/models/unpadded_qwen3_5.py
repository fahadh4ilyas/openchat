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
""" PyTorch Unpadded & Fused Qwen3_5 model. Compatible with HF. """

from typing import Optional, Tuple

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers import initialization as init
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel, Qwen3_5GatedDeltaNet, Qwen3_5RMSNorm, Qwen3_5VisionRotaryEmbedding

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

    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    output = (1 + weight.to(torch.float32)) * hidden_states
    return output.to(input_dtype)


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
    rotary_dim: int,
    fast_rope: bool = False,
):
    # q, k:     [nnz, num_heads, head_dim]
    # cos, sin: [nnz, rotary_dim]

    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    if fast_rope:
        q_embed, k_embed = fast_rope_embedding(q_rot, k_rot, cos, sin)
    else:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q = torch.cat((q_embed, q_pass), dim=-1)
    k = torch.cat((k_embed, k_pass), dim=-1)

    return q, k


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen3_5
class UnpaddedQwen3_5RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        """
        UnpaddedQwen3_5RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, use_fast_norm: bool = False):
        if use_fast_norm:
            return fast_rms_layernorm(hidden_states, self.weight, self.variance_epsilon, gemma=True)
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class UnpaddedQwen3_5RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        mrope_section=(11,11,10),
        device=None,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)

        # ensure even
        self.rotary_dim -= self.rotary_dim % 2

        self.mrope_section = mrope_section
        self.attention_scaling = 1.0

        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, self.rotary_dim, 2, device=device).float()
                / self.rotary_dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)


    def apply_interleaved_mrope(self, freqs):

        freqs_t = freqs[0]

        for dim, offset in enumerate((1,2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)

            freqs_t[..., idx] = freqs[dim, ..., idx]

        return freqs_t


    def forward(self, x, position_ids):
        """
        x : (nnz, heads, head_dim)
        position_ids:
            text → (nnz,)
            multimodal → (3, nnz)
        """

        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(3, -1)

        inv_freq = self.inv_freq.float()

        # compute frequencies
        freqs = position_ids[..., None].float() * inv_freq[None, None, :]

        freqs = self.apply_interleaved_mrope(freqs)

        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        return cos, sin


class UnpaddedQwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnpaddedQwen3_5Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.rotary_dim = int(self.head_dim * config.rope_parameters["partial_rotary_factor"])

        self.q_proj = nn.Linear(
            self.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
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
        self.q_norm = UnpaddedQwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = UnpaddedQwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        use_fast_rope: bool = False,
        use_fast_norm: bool = False,
    ) -> torch.Tensor:
        # nz_hidden_states: [nnz, num_heads, head_dim]
        # cu_seqlens:       [bs + 1]

        input_shape = nz_hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(self.q_proj(nz_hidden_states).view(
            *(*input_shape, -1, self.head_dim * 2)
        ), 2, dim=-1)
        query_states = self.q_norm(query_states, use_fast_norm)
        gate = gate.reshape(*(*input_shape, -1))
        key_states = self.k_norm(self.k_proj(nz_hidden_states).view(
            *hidden_shape
        ), use_fast_norm)
        value_states = self.v_proj(nz_hidden_states).view(
            *hidden_shape
        )

        # RoPE
        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rotary_dim, use_fast_rope
        )


        # flash attn
        if cu_seqlens[-1] == max_seqlen:
            attn_output = flash_attn_func(
                q=query_states.unsqueeze(0),
                k=key_states.unsqueeze(0),
                v=value_states.unsqueeze(0),
                softmax_scale=self.scaling,
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
                softmax_scale=self.scaling,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
            )

        # attn_output: [total_nnz, num_heads, head_dim]
        attn_output = attn_output.view(-1, self.num_attention_heads * self.head_dim) * torch.sigmoid(gate)  # type: ignore
        return self.o_proj(attn_output)


def unpack_padded_fast(hidden_states: torch.Tensor, cu_seqlens: torch.Tensor):

    device = hidden_states.device
    B = cu_seqlens.numel() - 1
    H = hidden_states.size(-1)

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    T_max = lengths.max()

    positions = torch.arange(T_max, device=device).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)

    starts = cu_seqlens[:-1].unsqueeze(1)

    gather = starts + positions

    flat = gather[mask]

    padded = hidden_states.new_zeros(B, T_max, H)

    padded[mask] = hidden_states[flat]

    return padded, mask


def pack_padded_fast(padded: torch.Tensor, mask: torch.Tensor):
    return padded[mask]


class UnpaddedQwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = UnpaddedQwen3_5Attention(config=config, layer_idx=layer_idx)
        self.mlp = UnpaddedQwen3_5MLP(config=config)
        self.input_layernorm = UnpaddedQwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = UnpaddedQwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        use_fast_norm: bool = False,
        use_fast_rope: bool = False,
    ) -> torch.Tensor:
        # Self Attention
        residual = nz_hidden_states

        nz_hidden_states = self.input_layernorm(nz_hidden_states, use_fast_norm)
        if self.layer_type == "linear_attention":
            hidden_states, attention_mask = unpack_padded_fast(
                nz_hidden_states, cu_seqlens
            )
            
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask
            )

            nz_hidden_states = pack_padded_fast(hidden_states, attention_mask)
        elif self.layer_type == "full_attention":
            nz_hidden_states = self.self_attn(
                cos_sin=cos_sin,
                nz_hidden_states=nz_hidden_states,
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


class UnpaddedQwen3_5PreTrainedModel(PreTrainedModel):
    config: Qwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _no_split_modules = ["UnpaddedQwen3_5DecoderLayer", "Qwen3_5VisionBlock"]
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _is_stateful = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_5GatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, (Qwen3_5RMSNorm, UnpaddedQwen3_5RMSNorm)):
            init.zeros_(module.weight)
        elif isinstance(module, Qwen3_5VisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class UnpaddedQwen3_5TextModel(UnpaddedQwen3_5PreTrainedModel):
    config: Qwen3_5TextConfig
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UnpaddedQwen3_5DecoderLayer`]

    Args:
        config: Qwen3_5TextConfig
    """

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.rotary_emb = UnpaddedQwen3_5RotaryEmbedding(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rope_theta=config.rope_parameters["rope_theta"],
            partial_rotary_factor=config.rope_parameters["partial_rotary_factor"],
            mrope_section=config.rope_parameters["mrope_section"]
        )

        self.layers = nn.ModuleList(
            [
                UnpaddedQwen3_5DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = UnpaddedQwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        nz_hidden_states: torch.Tensor,
        nz_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        use_fast_norm: bool = False,
        use_fast_rope: bool = False,
    ) -> torch.Tensor:
        cos_sin = self.rotary_emb(nz_hidden_states, nz_position_ids)

        # decoder layers
        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                nz_hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    cos_sin,
                    nz_hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    use_fast_norm,
                    use_fast_rope,
                )
            else:
                nz_hidden_states = decoder_layer(
                    cos_sin=cos_sin,
                    nz_hidden_states=nz_hidden_states,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    use_fast_norm=use_fast_norm,
                    use_fast_rope=use_fast_rope,
                )

        nz_hidden_states = self.norm(nz_hidden_states, use_fast_norm)

        return nz_hidden_states


def compute_packed_mrope_positions(
    nz_input_ids,
    cu_seqlens,
    mm_token_type_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    spatial_merge_size=1,
    device=None,
):
    """
    Returns:
        nz_position_ids : (3, nnz)
    """

    nnz = nz_input_ids.shape[0]
    pos_ids = torch.zeros(3, nnz, dtype=torch.long, device=device)

    image_iter = iter(image_grid_thw) if image_grid_thw is not None else None
    video_iter = iter(video_grid_thw) if video_grid_thw is not None else None

    for b in range(len(cu_seqlens) - 1):

        start = cu_seqlens[b].item()
        end   = cu_seqlens[b+1].item()

        if (nz_input_ids[start:end] == 0).all():
            continue

        seq_types = mm_token_type_ids[start:end]

        current_pos = 0
        i = 0

        while i < len(seq_types):

            token_type = seq_types[i].item()

            j = i
            while j < len(seq_types) and seq_types[j] == token_type:
                j += 1

            length = j - i

            # -------------------------
            # TEXT TOKENS
            # -------------------------
            if token_type == 0:

                t = torch.arange(
                    current_pos,
                    current_pos + length,
                    device=device
                )

                pos_ids[:, start+i:start+j] = t.unsqueeze(0).expand(3, -1)

                current_pos += length

            # -------------------------
            # IMAGE TOKENS
            # -------------------------
            elif token_type == 1:

                grid = next(image_iter)
                T, H, W = grid.tolist()

                H = H // spatial_merge_size
                W = W // spatial_merge_size

                t = torch.zeros(T*H*W, device=device, dtype=torch.long)
                h = torch.arange(H, device=device).repeat_interleave(W*T)
                w = torch.arange(W, device=device).repeat(H*T)

                pos_ids[0, start+i:start+j] = t + current_pos
                pos_ids[1, start+i:start+j] = h + current_pos
                pos_ids[2, start+i:start+j] = w + current_pos

                current_pos += max(H, W)

            # -------------------------
            # VIDEO TOKENS
            # -------------------------
            elif token_type == 2:

                grid = next(video_iter)
                T, H, W = grid.tolist()

                H = H // spatial_merge_size
                W = W // spatial_merge_size

                t = torch.arange(T, device=device).repeat_interleave(H*W)
                h = torch.arange(H, device=device).repeat_interleave(W*T)
                w = torch.arange(W, device=device).repeat(H*T)

                pos_ids[0, start+i:start+j] = t + current_pos
                pos_ids[1, start+i:start+j] = h + current_pos
                pos_ids[2, start+i:start+j] = w + current_pos

                current_pos += max(H, W)

            i = j

    return pos_ids


class UnpaddedQwen3_5Model(UnpaddedQwen3_5PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UnpaddedQwen3_5DecoderLayer`]

    Args:
        config: Qwen3_5Config
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)

        self.config = config
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = UnpaddedQwen3_5TextModel._from_config(config.text_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def forward(
        self,

        # packed tokens
        nz_input_ids: torch.LongTensor,
        cu_seqlens: torch.LongTensor,
        max_seqlen: int,

        # multimodal inputs
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,

        use_fast_norm: bool = False,
        use_fast_rope: bool = False,
    ):

        device = nz_input_ids.device

        # --------------------------------------------------
        # 1. TEXT EMBEDDINGS
        # --------------------------------------------------

        nz_hidden_states = self.language_model.get_input_embeddings()(nz_input_ids)

        # --------------------------------------------------
        # 2. BUILD mm_token_type_ids
        # --------------------------------------------------

        mm_token_type_ids = torch.zeros_like(nz_input_ids)

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id

        if pixel_values is not None:
            mm_token_type_ids[nz_input_ids == image_token_id] = 1

        if pixel_values_videos is not None:
            mm_token_type_ids[nz_input_ids == video_token_id] = 2

        # --------------------------------------------------
        # 3. IMAGE FEATURES
        # --------------------------------------------------

        if pixel_values is not None:

            image_outputs = self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                return_dict=True,
            )

            image_embeds = image_outputs.pooler_output.to(
                nz_hidden_states.device,
                nz_hidden_states.dtype,
            )

            image_mask = nz_input_ids == image_token_id

            if image_mask.sum() != image_embeds.shape[0]:
                raise ValueError(
                    f"Image placeholder mismatch: "
                    f"{image_mask.sum()} tokens vs {image_embeds.shape[0]} features"
                )

            nz_hidden_states[image_mask] = image_embeds

        # --------------------------------------------------
        # 4. VIDEO FEATURES
        # --------------------------------------------------

        if pixel_values_videos is not None:

            video_outputs = self.visual(
                pixel_values_videos,
                grid_thw=video_grid_thw,
                return_dict=True,
            )

            video_embeds = video_outputs.pooler_output.to(
                nz_hidden_states.device,
                nz_hidden_states.dtype,
            )

            video_mask = nz_input_ids == video_token_id

            if video_mask.sum() != video_embeds.shape[0]:
                raise ValueError(
                    f"Video placeholder mismatch: "
                    f"{video_mask.sum()} tokens vs {video_embeds.shape[0]} features"
                )

            nz_hidden_states[video_mask] = video_embeds

        # --------------------------------------------------
        # 5. COMPUTE PACKED 3D POSITION IDS
        # --------------------------------------------------

        nz_position_ids = compute_packed_mrope_positions(
            nz_input_ids=nz_input_ids,
            cu_seqlens=cu_seqlens,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            device=device,
        )

        # --------------------------------------------------
        # 6. LANGUAGE MODEL
        # --------------------------------------------------

        nz_hidden_states = self.language_model(
            nz_hidden_states=nz_hidden_states,
            nz_position_ids=nz_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            use_fast_norm=use_fast_norm,
            use_fast_rope=use_fast_rope,
        )

        return nz_hidden_states


class Qwen3_5ForConditionalGeneration(UnpaddedQwen3_5PreTrainedModel):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.model = UnpaddedQwen3_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

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
        # multimodal inputs
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
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
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_fast_norm=use_fast_norm,
            use_fast_rope=use_fast_rope,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if nz_shifted_label_ids is not None:
            assert nz_shifted_loss_weights is not None

            if num_seq > 0:
                acc = (
                    weighted_token_accuracy(
                        logits.detach(), nz_shifted_label_ids, nz_shifted_loss_weights
                    )
                    / num_seq
                )
                loss = (
                    weighted_cross_entropy(
                        logits, nz_shifted_label_ids, nz_shifted_loss_weights
                    )
                    / num_seq,
                    acc,
                )
            else:
                acc = weighted_token_accuracy(
                    logits.detach(), nz_shifted_label_ids, nz_shifted_loss_weights
                )
                loss = (
                    weighted_cross_entropy(
                        logits, nz_shifted_label_ids, nz_shifted_loss_weights
                    ),
                    acc,
                )

        return CausalLMOutputWithPast(
            loss=loss,  # type: ignore
            logits=logits,
        )
