import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange, repeat
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from .rope_real import RotaryPosEmbedReal

try:
    from transformers.modeling_flash_attention_utils import (  # type: ignore
        flash_attn_varlen_func,  # pyright: ignore[reportAttributeAccessIssue]
        is_flash_attn_available,
    )
except Exception:  # pragma: no cover - best-effort compatibility
    flash_attn_varlen_func = None  # type: ignore[assignment]

    def is_flash_attn_available() -> bool:  # type: ignore[override]
        return False


from .rope_real import apply_real_rotary_emb

_HAS_FLASH_ATTN_VARLEN = bool(is_flash_attn_available()) and flash_attn_varlen_func is not None

logger = init_logger(__name__)


# =============================================================================
# TP Constraint Validation
# =============================================================================


def _positive_divisors(n: int) -> set[int]:
    """Return all positive divisors of n."""
    if n <= 0:
        return set()
    divs: set[int] = set()
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            divs.add(d)
            divs.add(n // d)
    return divs


def validate_mammothmoda2_tp_constraints(
    *,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    ffn_inner_dim: int,
    tensor_parallel_size: int,
) -> list[int]:
    """Validate MammothModa2 TP constraints.

    Required constraints:
    - dim % tp_size == 0
    - num_heads % tp_size == 0
    - num_kv_heads % tp_size == 0
    - ffn_inner_dim % tp_size == 0

    Returns:
        List of supported TP sizes.
    """
    tp_size = int(tensor_parallel_size)
    if tp_size <= 0:
        raise ValueError(f"tensor_parallel_size must be > 0, got {tp_size}")

    errors = []

    if dim % tp_size != 0:
        supported = sorted(_positive_divisors(dim))
        errors.append(f"dim ({dim}) must be divisible by tp_size ({tp_size}). Supported: {supported}")

    if num_heads % tp_size != 0:
        supported = sorted(_positive_divisors(num_heads))
        errors.append(f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size}). Supported: {supported}")

    if num_kv_heads % tp_size != 0:
        supported = sorted(_positive_divisors(num_kv_heads))
        errors.append(f"num_kv_heads ({num_kv_heads}) must be divisible by tp_size ({tp_size}). Supported: {supported}")

    if ffn_inner_dim % tp_size != 0:
        supported = sorted(_positive_divisors(ffn_inner_dim))
        errors.append(f"ffn_inner_dim ({ffn_inner_dim}) must be divisible by tp_size ({tp_size}). Supported: {supported}")

    if errors:
        raise ValueError("MammothModa2 TP constraint violations:\n" + "\n".join(f"  - {e}" for e in errors))

    supported_tp = sorted(
        _positive_divisors(num_heads)
        & _positive_divisors(num_kv_heads)
        & _positive_divisors(dim)
        & _positive_divisors(ffn_inner_dim)
    )
    return supported_tp


def _get_tp_size() -> int:
    """Get current tensor parallel size."""
    return get_tensor_model_parallel_world_size()


# =============================================================================
# Layers
# =============================================================================


class LuminaRMSNormZero(nn.Module):
    """
    Norm layer adaptive RMS normalization zero.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_eps: float,
        norm_elementwise_affine: bool,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        # Modulation layer: kept replicated for precision sensitivity
        self.linear = nn.Linear(min(embedding_dim, 1024), 4 * embedding_dim, bias=True)
        self.norm = Qwen2RMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaFeedForward(nn.Module):
    """
    SwiGLU FeedForward with Tensor Parallel support.

    Original structure:
        linear_1: gate projection (dim -> inner_dim)
        linear_3: up projection (dim -> inner_dim)
        linear_2: down projection (inner_dim -> dim)

    TP structure (when tp_size > 1):
        w13: MergedColumnParallelLinear combining gate + up
        w2: RowParallelLinear for down (with all-reduce)
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: int | None = 256,
        ffn_dim_multiplier: float | None = None,
    ):
        super().__init__()

        # Apply multiplier
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        # Round up to multiple_of
        if multiple_of is not None:
            inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        tp_size = _get_tp_size()

        if tp_size > 1:
            # TP mode: use parallel layers
            self.w13 = MergedColumnParallelLinear(
                dim,
                [inner_dim, inner_dim],  # [gate_dim, up_dim]
                bias=False,
                return_bias=False,
            )
            self.act = SiluAndMul()
            self.w2 = RowParallelLinear(
                inner_dim,
                dim,
                bias=False,
                input_is_parallel=True,
                return_bias=False,
            )
            self._is_tp = True
        else:
            # Non-TP mode: use standard layers
            self.linear_1 = nn.Linear(dim, inner_dim, bias=False)  # gate
            self.linear_2 = nn.Linear(inner_dim, dim, bias=False)  # down
            self.linear_3 = nn.Linear(dim, inner_dim, bias=False)  # up
            self._is_tp = False

    def swiglu(self, x, y):
        return F.silu(x.float(), inplace=False).to(x.dtype) * y

    def forward(self, x):
        if self._is_tp:
            # TP path
            x = self.w13(x)
            x = self.act(x)
            return self.w2(x)
        else:
            # Non-TP path
            h1, h2 = self.linear_1(x), self.linear_3(x)
            return self.linear_2(self.swiglu(h1, h2))


class LuminaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim: int | None = None,
    ):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = Qwen2RMSNorm(embedding_dim, eps=eps)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.linear_2 = None
        if out_dim is not None:
            self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        x = self.norm(x) * (1 + scale)[:, None, :]

        if self.linear_2 is not None:
            x = self.linear_2(x)

        return x


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2520,
        text_feat_dim: int = 3584,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=timestep_scale
        )

        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024)
        )

        self.caption_embedder = nn.Sequential(
            Qwen2RMSNorm(text_feat_dim, eps=norm_eps),
            nn.Linear(text_feat_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).to(dtype=dtype)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(text_hidden_states)
        return time_embed, caption_embed


class SimpleQFormerImageRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 128,
        num_layers: int = 2,
        num_heads: int | None = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        # ensure num_heads divides hidden_size
        if num_heads is None:
            num_heads = max(1, hidden_size // 128)
        self.num_heads = self._choose_valid_num_heads(hidden_size, num_heads)
        self.input_proj = nn.Sequential(
            Qwen2RMSNorm(hidden_size, eps=norm_eps),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # Learnable query embeddings
        scale = hidden_size**-0.5
        self.query = nn.Parameter(scale * torch.randn(1, num_queries, hidden_size))

        # Decoder layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    dict(
                        ln_q1=Qwen2RMSNorm(hidden_size, eps=norm_eps),
                        self_attn=nn.MultiheadAttention(
                            embed_dim=hidden_size, num_heads=self.num_heads, dropout=dropout, batch_first=True
                        ),
                        ln_q2=Qwen2RMSNorm(hidden_size, eps=norm_eps),
                        cross_attn=nn.MultiheadAttention(
                            embed_dim=hidden_size, num_heads=self.num_heads, dropout=dropout, batch_first=True
                        ),
                        ln_ffn=Qwen2RMSNorm(hidden_size, eps=norm_eps),
                        ffn=nn.Sequential(
                            nn.Linear(hidden_size, 4 * hidden_size, bias=False),
                            nn.SiLU(),
                            nn.Linear(4 * hidden_size, hidden_size, bias=False),
                        ),
                    )
                )
            )

    @staticmethod
    def _choose_valid_num_heads(hidden_size: int, proposed_heads: int, preferred_head_dim: int = 128) -> int:
        """Pick a number of heads that divides hidden_size, close to proposed or preferred."""
        # If proposed is valid, use it
        if proposed_heads > 0 and hidden_size % proposed_heads == 0:
            return proposed_heads
        # target based on preferred head dim
        target = max(1, round(hidden_size / preferred_head_dim))
        # collect divisors up to 128 heads (more than enough)
        max_heads_cap = min(128, hidden_size)
        divisors = [d for d in range(1, max_heads_cap + 1) if hidden_size % d == 0]
        # choose closest to target
        best = min(divisors, key=lambda d: (abs(d - target), -d))
        return best

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tensor of shape (batch, num_queries, hidden_size)
        """
        batch, _, _ = x.shape
        kv = self.input_proj(x)
        q = self.query.repeat(batch, 1, 1).to(kv.dtype)

        for layer in self.layers:
            # Self-attention on queries
            q_norm = layer["ln_q1"](q)
            attn_out, _ = layer["self_attn"](q_norm, q_norm, q_norm, need_weights=False)
            q = q + attn_out

            # Cross-attention: queries attend to inputs
            q_norm = layer["ln_q2"](q)
            cross_out, _ = layer["cross_attn"](q_norm, kv, kv, need_weights=False, key_padding_mask=attention_mask)
            q = q + cross_out

            # Feed-forward
            q = q + layer["ffn"](layer["ln_ffn"](q))

        return q


class AttnProcessor:
    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor requires PyTorch 2.0+ (F.scaled_dot_product_attention).")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        base_sequence_length: int | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if image_rotary_emb is not None:
            query = apply_real_rotary_emb(query, image_rotary_emb[0], image_rotary_emb[1])
            key = apply_real_rotary_emb(key, image_rotary_emb[0], image_rotary_emb[1])

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        if _HAS_FLASH_ATTN_VARLEN and attention_mask is not None and hidden_states.is_cuda:
            # Flash-Attn varlen expects packed tokens + cu_seqlens. Here we only need
            # the self-attention case (q/k/v share the same padding mask).
            attention_mask = attention_mask.to(torch.bool)
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen = int(seqlens.max().item())
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

            query_states = query.reshape(batch_size * sequence_length, attn.heads, head_dim)[indices]
            key_states = key.reshape(batch_size * sequence_length, kv_heads, head_dim)[indices]
            value_states = value.reshape(batch_size * sequence_length, kv_heads, head_dim)[indices]

            if kv_heads < attn.heads:
                key_states = repeat(key_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)
                value_states = repeat(value_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=False,
                softmax_scale=softmax_scale,
            )

            out = torch.zeros(
                (batch_size * sequence_length, attn.heads, head_dim),
                device=attn_output_unpad.device,
                dtype=attn_output_unpad.dtype,
            )
            out[indices] = attn_output_unpad
            hidden_states = out.view(batch_size, sequence_length, attn.heads, head_dim).flatten(-2)
            hidden_states = hidden_states.type_as(query)
        else:
            # PyTorch SDPA path.
            attn_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.bool)
                attn_mask = attention_mask.view(batch_size, 1, 1, -1)

            query = query.transpose(1, 2)  # [B, H, S, D]
            key = key.transpose(1, 2)  # [B, H_kv, S, D]
            value = value.transpose(1, 2)

            if kv_heads < attn.heads:
                key = key.repeat_interleave(attn.heads // kv_heads, dim=1)
                value = value.repeat_interleave(attn.heads // kv_heads, dim=1)

            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=softmax_scale,
            )

            if attention_mask is not None:
                # Keep padding tokens consistent with the flash-varlen path (zero output).
                hidden_states = hidden_states * attention_mask[:, None, :, None]

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


# Import diffusers Attention for non-TP mode
from diffusers.models.attention_processor import Attention


class TPAttention(nn.Module):
    """
    Tensor Parallel Attention for MammothModa2.

    Uses QKVParallelLinear for Q/K/V projections and RowParallelLinear for output.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        # QKV projection (Column Parallel)
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
        )

        # QK normalization
        self.norm_q = Qwen2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Qwen2RMSNorm(self.head_dim, eps=eps)

        # Output projection (Row Parallel)
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    input_is_parallel=True,
                    return_bias=False,
                ),
                nn.Identity(),  # placeholder for dropout
            ]
        )

        # Attention layer
        from vllm_omni.diffusion.attention.layer import Attention as VLLMAttention

        self.attn = VLLMAttention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv, _ = self.to_qkv(hidden_states)

        # Split Q, K, V using LOCAL head counts
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape: [B, S, H, D]
        query = query.view(batch_size, seq_len, self.to_qkv.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.to_qkv.num_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.to_qkv.num_kv_heads, self.head_dim)

        # QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE if provided
        if image_rotary_emb is not None:
            query = apply_real_rotary_emb(query, image_rotary_emb[0], image_rotary_emb[1])
            key = apply_real_rotary_emb(key, image_rotary_emb[0], image_rotary_emb[1])

        query, key = query.to(hidden_states.dtype), key.to(hidden_states.dtype)

        # Attention
        hidden_states = self.attn(query, key, value)

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)

        # Output projection
        hidden_states = self.to_out[0](hidden_states)

        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        """Initialize the transformer block."""
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads

        tp_size = _get_tp_size()

        if tp_size > 1:
            # TP mode: use parallel attention
            self.attn = TPAttention(
                dim=dim,
                num_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                eps=1e-5,
            )
            self._is_tp_attn = True
        else:
            # Non-TP mode: use diffusers Attention
            processor = AttnProcessor()
            self.attn = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                qk_norm=None,
                heads=num_attention_heads,
                kv_heads=num_kv_heads,
                eps=1e-5,
                bias=False,
                out_bias=False,
                processor=processor,
            )
            self.attn.norm_q = Qwen2RMSNorm(self.head_dim, eps=1e-5)
            self.attn.norm_k = Qwen2RMSNorm(self.head_dim, eps=1e-5)
            self._is_tp_attn = False

        # Initialize feed-forward network (TP-enabled in LuminaFeedForward)
        self.feed_forward = LuminaFeedForward(
            dim=dim, inner_dim=4 * dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier
        )

        # Initialize normalization layers
        if modulation:
            self.norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True)
        else:
            self.norm1 = Qwen2RMSNorm(dim, eps=norm_eps)

        self.ffn_norm1 = Qwen2RMSNorm(dim, eps=norm_eps)
        self.norm2 = Qwen2RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = Qwen2RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            if temb is None:
                raise ValueError("temb must be provided when modulation is enabled")

            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

            if self._is_tp_attn:
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
            else:
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

            hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
            hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        else:
            norm_hidden_states = self.norm1(hidden_states)

            if self._is_tp_attn:
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
            else:
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

            hidden_states = hidden_states + self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class Transformer2DModel(ModelMixin, ConfigMixin):
    """MammothModa2 DiT transformer with Tensor Parallel support."""

    # For HSDP sharding
    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return (
            ("layers" in name or "noise_refiner" in name or "context_refiner" in name or "ref_image_refiner" in name)
            and name.split(".")[-1].isdigit()
        )

    _hsdp_shard_conditions = [_is_transformer_block]

    # Weight mapping for loading original (non-TP) weights into TP model
    packed_modules_mapping = {
        "feed_forward.w13": ["feed_forward.linear_1", "feed_forward.linear_3"],
        "attn.to_qkv": ["attn.to_q", "attn.to_k", "attn.to_v"],
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int | None = None,
        hidden_size: int = 2304,
        num_layers: int = 26,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: tuple[int, int, int] = (32, 32, 32),
        axes_lens: tuple[int, int, int] = (300, 512, 512),
        text_feat_dim: int = 1024,
        timestep_scale: float = 1.0,
    ) -> None:
        """Initialize the  transformer model."""
        super().__init__()
        self.hidden_size = hidden_size

        # Validate configuration
        if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
            raise ValueError(
                f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
            )

        self.out_channels = out_channels or in_channels

        # Calculate FFN inner dimension
        ffn_inner_dim = 4 * hidden_size
        if ffn_dim_multiplier is not None:
            ffn_inner_dim = int(ffn_dim_multiplier * ffn_inner_dim)
        ffn_inner_dim = multiple_of * ((ffn_inner_dim + multiple_of - 1) // multiple_of)

        # Validate TP constraints
        tp_size = _get_tp_size()
        supported_tp = validate_mammothmoda2_tp_constraints(
            dim=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            ffn_inner_dim=ffn_inner_dim,
            tensor_parallel_size=tp_size,
        )

        logger.info(
            "MammothModa2 init: dim=%d num_heads=%d num_kv_heads=%d ffn_inner_dim=%d tp=%d (supported=%s)",
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            ffn_inner_dim,
            tp_size,
            tuple(supported_tp),
        )

        # Initialize embeddings
        self.rope_embedder = RotaryPosEmbedReal(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.ref_image_patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            text_feat_dim=text_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale,
        )

        # Initialize transformer blocks
        self.noise_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.ref_image_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=False,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # 3. Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        # Add learnable embeddings to distinguish different images
        self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size))  # support max 5 ref images

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        ref_image_hidden_states: list[list[torch.Tensor]] | None,
        return_dict: bool,
    ) -> tuple[int, int, int]:
        if return_dict:
            raise ValueError("return_dict=True is not supported in vLLM inference.")
        if ref_image_hidden_states is not None:
            raise ValueError("ref_image_hidden_states is not supported in vLLM inference.")
        if hidden_states.ndim != 4:
            raise ValueError(f"Expected hidden_states to be 4D [B,C,H,W], got shape={tuple(hidden_states.shape)}")

        batch_size, _channels, height, width = hidden_states.shape
        if batch_size != text_hidden_states.shape[0] or batch_size != text_attention_mask.shape[0]:
            raise ValueError(
                "Batch size mismatch: "
                f"hidden_states={batch_size}, text_hidden_states={text_hidden_states.shape[0]}, "
                f"text_attention_mask={text_attention_mask.shape[0]}"
            )

        p = self.config.patch_size
        if height % p != 0 or width % p != 0:
            raise ValueError(f"Input latent H/W must be divisible by patch_size={p}, got {height}x{width}")
        return batch_size, height, width

    def _prepare_embeddings(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
    ):
        device = hidden_states.device
        p = self.config.patch_size

        temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states.dtype)

        img_tokens = rearrange(hidden_states, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        img_tokens = self.x_embedder(img_tokens)

        img_len = (height // p) * (width // p)
        img_mask = torch.ones((batch_size, img_len), dtype=torch.bool, device=device)
        l_effective_img_len = [img_len for _ in range(batch_size)]
        img_sizes = [(height, width) for _ in range(batch_size)]

        l_effective_ref_img_len = [[] for _ in range(batch_size)]
        ref_img_sizes = [None for _ in range(batch_size)]

        (
            context_rotary_emb,
            _ref_img_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = self.rope_embedder(
            freqs_cis,
            text_attention_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        return (
            temb,
            text_hidden_states,
            img_tokens,
            img_mask,
            img_len,
            context_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        )

    def _apply_refiners(
        self,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        context_rotary_emb: torch.Tensor,
        img_tokens: torch.Tensor,
        img_mask: torch.Tensor,
        noise_rotary_emb: torch.Tensor,
        temb: torch.Tensor,
    ):
        for layer in self.context_refiner:
            text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)

        for layer in self.noise_refiner:
            img_tokens = layer(img_tokens, img_mask, noise_rotary_emb, temb)

        return text_hidden_states, img_tokens

    def _apply_transformer_layers(self, hidden_states, attention_mask, rotary_emb, temb):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        text_attention_mask: torch.Tensor,
        ref_image_hidden_states: list[list[torch.Tensor]] | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        batch_size, height, width = self._validate_inputs(
            hidden_states, text_hidden_states, text_attention_mask, ref_image_hidden_states, return_dict
        )

        (
            temb,
            text_hidden_states,
            img_tokens,
            img_mask,
            img_len,
            context_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = self._prepare_embeddings(
            hidden_states,
            timestep,
            text_hidden_states,
            text_attention_mask,
            freqs_cis,
            batch_size,
            height,
            width,
        )

        text_hidden_states, img_tokens = self._apply_refiners(
            text_hidden_states,
            text_attention_mask,
            context_rotary_emb,
            img_tokens,
            img_mask,
            noise_rotary_emb,
            temb,
        )

        max_seq_len = max(seq_lengths)
        attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.config.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            attention_mask[i, :seq_len] = True
            joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len : encoder_seq_len + img_len] = img_tokens[i, :img_len]

        hidden_states = self._apply_transformer_layers(joint_hidden_states, attention_mask, rotary_emb, temb)

        hidden_states = self.norm_out(hidden_states, temb)

        p = self.config.patch_size
        img_hidden_states = torch.stack(
            [
                hidden_states[i, encoder_seq_len : encoder_seq_len + img_len]
                for i, encoder_seq_len in enumerate(encoder_seq_lengths)
            ],
            dim=0,
        )
        output = rearrange(
            img_hidden_states,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=height // p,
            w=width // p,
            p1=p,
            p2=p,
        )
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with TP-aware mapping.

        Handles weight mapping for TP-sharded layers:
        - feed_forward.linear_1 + feed_forward.linear_3 -> feed_forward.w13
        - attn.to_q + attn.to_k + attn.to_v -> attn.to_qkv (TP mode only)
        """
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        # Stacked params mapping for TP layers
        # (param_name, weight_name, shard_id)
        stacked_params_mapping = [
            # FFN: gate + up -> w13
            (".feed_forward.w13.", ".feed_forward.linear_1.", 0),
            (".feed_forward.w13.", ".feed_forward.linear_3.", 1),
        ]

        # For TP attention, add QKV mapping
        if _get_tp_size() > 1:
            stacked_params_mapping.extend([
                # Attention: Q, K, V -> to_qkv
                (".attn.to_qkv.", ".attn.to_q.", "q"),
                (".attn.to_qkv.", ".attn.to_k.", "k"),
                (".attn.to_qkv.", ".attn.to_v.", "v"),
            ])

        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())
        loaded_params = set[str]()

        for name, loaded_weight in weights:
            # Skip LLM weights if any
            if name.startswith("llm_model."):
                continue

            # Check for stacked params
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    # Replace weight name with param name
                    new_name = name.replace(weight_name, param_name)
                    if new_name in params_dict:
                        param = params_dict[new_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_params.add(new_name)
                    break
            else:
                # Regular weight loading
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params
