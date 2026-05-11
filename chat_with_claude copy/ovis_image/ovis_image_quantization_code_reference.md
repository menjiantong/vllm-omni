# OvisImage 量化实现代码参考

本文档提供完整的代码修改参考，可直接用于实现。

---

## 1. 完整修改后的 ovis_image_transformer.py

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 Alibaba Ovis-Image Team and The HuggingFace. All rights reserved.
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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_npu_available
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.adalayernorm import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)


class OvisImageAttention(nn.Module):
    """OvisImage 注意力模块，支持量化。"""

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # 修改: 添加 quant_config 和 prefix，移除 disable_tp
        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList([
                RowParallelLinear(
                    self.inner_dim,
                    self.out_dim,
                    bias=out_bias,
                    input_is_parallel=True,
                    return_bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                ),
                nn.Dropout(dropout),
            ])

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=self.added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                bias=added_proj_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.add_kv_proj",
            )

            self.to_add_out = RowParallelLinear(
                self.inner_dim,
                query_dim,
                bias=out_bias,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.to_add_out",
            )

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # FP8 量化需要 contiguous 输入
        hidden_states = hidden_states.contiguous()
        qkv, _ = self.to_qkv(hidden_states)

        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if self.added_kv_proj_dim is not None:
            encoder_hidden_states = encoder_hidden_states.contiguous()
            encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
            add_q_size = self.add_kv_proj.num_heads * self.head_dim
            add_kv_size = self.add_kv_proj.num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )

            encoder_query = encoder_query.unflatten(-1, (self.add_kv_proj.num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)

        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            # FP8 RowParallelLinear 需要 contiguous 输入
            hidden_states = self.to_out[0](hidden_states.contiguous())
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states.contiguous())

            return hidden_states, encoder_hidden_states
        else:
            if get_tensor_model_parallel_world_size() > 1:
                hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states


class OvisImageSingleTransformerBlock(nn.Module):
    """OvisImage 单流 Transformer 块，支持量化。"""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # 调制层保持全精度
        self.norm = AdaLayerNormZeroSingle(dim, quant_config=None, prefix=f"{prefix}.norm")
        self.proj_mlp = ReplicatedLinear(
            dim,
            self.mlp_hidden_dim * 2,
            bias=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj_mlp",
        )
        self.act_mlp = nn.SiLU()
        self.proj_out = ReplicatedLinear(
            dim + self.mlp_hidden_dim,
            dim,
            bias=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj_out",
        )

        self.attn = OvisImageAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states, mlp_hidden_gate = torch.split(
            self.proj_mlp(norm_hidden_states), [self.mlp_hidden_dim, self.mlp_hidden_dim], dim=-1
        )
        mlp_hidden_states = self.act_mlp(mlp_hidden_gate) * mlp_hidden_states
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


class OvisImageTransformerBlock(nn.Module):
    """OvisImage 双流 Transformer 块，支持量化。"""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()

        # 调制层保持全精度
        self.norm1 = AdaLayerNormZero(dim, quant_config=None, prefix=f"{prefix}.norm1")
        self.norm1_context = AdaLayerNormZero(dim, quant_config=None, prefix=f"{prefix}.norm1_context")

        self.attn = OvisImageAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = OvisImageFeedForward(dim=dim, dim_out=dim, quant_config=quant_config, prefix=f"{prefix}.ff")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = OvisImageFeedForward(dim=dim, dim_out=dim, quant_config=quant_config, prefix=f"{prefix}.ff_context")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class OvisImageFeedForward(nn.Module):
    """支持量化的 SwiGLU FeedForward 层。"""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()

        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # SwiGLU: gate * up_proj
        self.gate_up_proj = ColumnParallelLinear(
            dim,
            inner_dim * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            inner_dim,
            dim_out,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        if isinstance(x, tuple):
            x = x[0]
        gate, up = x.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)[0]


class OvisImagePosEmbed(nn.Module):
    """3D 旋转位置编码。"""

    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            freqs_cis = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                use_real=False,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class OvisImageTransformer2DModel(nn.Module):
    """
    The Transformer model introduced in Ovis-Image.

    Reference: https://github.com/AIDC-AI/Ovis-Image

    支持量化配置，推荐仅量化 single_transformer_blocks。
    """

    _repeated_blocks = ["OvisImageTransformerBlock", "OvisImageSingleTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]

    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = 64,
        num_layers: int = 6,
        num_single_layers: int = 27,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 2048,
        axes_dims_rope: tuple[int] = (16, 56, 56),
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        model_config = od_config.tf_model_config
        num_layers = model_config.num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = OvisImagePosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)

        self.context_embedder_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.context_embedder = ReplicatedLinear(
            joint_attention_dim,
            self.inner_dim,
            quant_config=quant_config,
            prefix="context_embedder",
        )
        self.x_embedder = ReplicatedLinear(
            in_channels,
            self.inner_dim,
            quant_config=quant_config,
            prefix="x_embedder",
        )

        # 双流块: 建议保持全精度（参考 Flux #2728）
        self.transformer_blocks = nn.ModuleList(
            [
                OvisImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=None,  # 保持全精度
                    prefix=f"transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )

        # 单流块: 可以量化以节省显存
        self.single_transformer_blocks = nn.ModuleList(
            [
                OvisImageSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=quant_config,
                    prefix=f"single_transformer_blocks.{i}",
                )
                for i in range(num_single_layers)
            ]
        )

        # 输出调制层保持全精度
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            quant_config=None,
            prefix="norm_out",
        )
        self.proj_out = ReplicatedLinear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
            quant_config=quant_config,
            prefix="proj_out",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        hidden_states = self.x_embedder(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

        timesteps_proj = self.time_proj(timestep)
        temb = self.timestep_embedder(timesteps_proj.to(device=hidden_states.device, dtype=hidden_states.dtype))

        encoder_hidden_states = self.context_embedder_norm(encoder_hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if isinstance(encoder_hidden_states, tuple):
            encoder_hidden_states = encoder_hidden_states[0]

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        if isinstance(output, tuple):
            output = output[0]

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]
        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())

        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps") or name.endswith(".weight_scale"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    # 尝试映射到 to_out
                    if ".to_out.0." in name:
                        name = name.replace(".to_out.0.", ".to_out.")
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
        return loaded_params
```

---

## 2. pipeline_ovis_image.py 修改

只需要修改一处，传递 `quant_config`:

```python
# 第 182 行附近

# 修改前:
self.transformer = OvisImageTransformer2DModel(od_config=od_config)

# 修改后:
self.transformer = OvisImageTransformer2DModel(
    od_config=od_config,
    quant_config=od_config.quantization_config,
)
```

---

## 3. 配置文件示例

### 3.1 部署配置 (YAML)

```yaml
# ovis_image_fp8.yaml
model: "AIDC-AI/Ovis-Image-7B"
tensor_parallel_size: 1
quantization: "fp8"

# 或选择性量化
quantization:
  method: "fp8"
  ignored_layers:
    - "transformer_blocks.*"
    - "norm_out.*"
```

### 3.2 INT8 在线量化配置

```yaml
# ovis_image_int8.yaml
model: "AIDC-AI/Ovis-Image-7B"
tensor_parallel_size: 1
quantization:
  method: "int8"
  activation_scheme: "dynamic"
```

### 3.3 API 调用示例

```python
from vllm_omni import DiffusionEngine

# FP8 量化
engine = DiffusionEngine(
    model="AIDC-AI/Ovis-Image-7B",
    quantization="fp8",
)

# 生成图像
output = engine.generate(
    prompt="a beautiful sunset over mountains",
    height=1024,
    width=1024,
)
```

---

## 4. 验证测试代码

```python
import torch
from vllm_omni.diffusion.models.ovis_image import OvisImageTransformer2DModel, OvisImagePipeline
from vllm_omni.diffusion.data import OmniDiffusionConfig

def test_quantization_config_propagation():
    """测试量化配置正确传播到各层"""
    config = OmniDiffusionConfig(
        model="test_model",
        quantization_config="fp8",
    )
    
    model = OvisImageTransformer2DModel(
        od_config=config,
        quant_config=config.quantization_config,
    )
    
    # 验证双流块不量化
    for block in model.transformer_blocks:
        assert block.attn.to_qkv.quant_config is None, "双流块不应量化"
        assert block.ff.gate_up_proj.quant_config is None, "双流块 FFN 不应量化"
    
    # 验证单流块量化
    for block in model.single_transformer_blocks:
        assert block.attn.to_qkv.quant_config is not None, "单流块应量化"
        assert block.proj_mlp.quant_config is not None, "单流块 MLP 应量化"
    
    print("✓ 量化配置传播测试通过")

def test_forward_pass():
    """测试量化模型前向传播"""
    config = OmniDiffusionConfig(
        model="test_model",
        quantization_config="fp8",
    )
    
    model = OvisImageTransformer2DModel(
        od_config=config,
        quant_config=config.quantization_config,
    ).cuda()
    
    # 模拟输入
    batch_size = 1
    img_seq = 1024
    txt_seq = 256
    
    hidden_states = torch.randn(batch_size, img_seq, 64, device="cuda", dtype=torch.float16)
    encoder_hidden_states = torch.randn(batch_size, txt_seq, 2048, device="cuda", dtype=torch.float16)
    timestep = torch.tensor([0.5], device="cuda", dtype=torch.float16)
    img_ids = torch.zeros(img_seq, 3, device="cuda", dtype=torch.float16)
    txt_ids = torch.zeros(txt_seq, 3, device="cuda", dtype=torch.float16)
    
    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )
    
    assert output.sample.shape == (batch_size, img_seq, 64), f"输出形状错误: {output.sample.shape}"
    print(f"✓ 前向传播测试通过，输出形状: {output.sample.shape}")

if __name__ == "__main__":
    test_quantization_config_propagation()
    test_forward_pass()
```

---

## 5. 修改清单总结

| 文件 | 修改内容 | 代码行数 |
|------|----------|----------|
| `ovis_image_transformer.py` | 添加量化支持、修改 Linear 层类型、添加 prefix 参数 | ~100 行修改 |
| `pipeline_ovis_image.py` | 传递 quant_config | 1 行修改 |

**预计工作量**: 1-2 天开发 + 1 天测试验证
