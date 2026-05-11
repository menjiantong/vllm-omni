# OvisImage 模型量化功能设计方案

## 1. 概述

本文档详细描述为 OvisImage 模型添加量化支持的设计方案。通过参考 vllm-omni 中 Flux 等类似架构的量化实现，制定一套完整的量化策略。

---

## 2. 当前代码分析

### 2.1 OvisImage 当前状态

当前 `OvisImageTransformer2DModel` 的量化支持存在以下问题:

```python
# ovis_image_transformer.py 第 73-79 行
self.to_qkv = QKVParallelLinear(
    hidden_size=query_dim,
    head_size=self.head_dim,
    total_num_heads=self.heads,
    disable_tp=True,  # 问题: 禁用了 TP，但没有传递 quant_config
    bias=bias,
)

# 第 90-96 行
self.add_kv_proj = QKVParallelLinear(
    hidden_size=self.added_kv_proj_dim,
    head_size=self.head_dim,
    total_num_heads=self.heads,
    disable_tp=True,  # 同样的问题
    bias=added_proj_bias,
)
```

**问题总结**:
1. 所有 Linear 层都没有传递 `quant_config` 参数
2. 使用了 `disable_tp=True`，这在量化场景下可能导致问题
3. 没有 `prefix` 参数用于权重加载和量化层识别
4. `OvisImageAttention` 没有接收 `quant_config` 参数

### 2.2 Flux 参考实现

Flux 模型的量化实现是 OvisImage 的最佳参考:

```python
# flux_transformer.py 第 149-156 行
self.to_qkv = QKVParallelLinear(
    hidden_size=query_dim,
    head_size=self.head_dim,
    total_num_heads=self.heads,
    bias=bias,
    quant_config=quant_config,  # ✓ 传递了 quant_config
    prefix=f"{prefix}.to_qkv",   # ✓ 传递了 prefix
)
```

**Flux 的量化策略**:
```python
# flux_transformer.py 第 571-581 行
# 双流块保持全精度 - FP8 在联合注意力路径上会产生噪声
self.transformer_blocks = nn.ModuleList(
    [
        FluxTransformerBlock(
            dim=self.inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            quant_config=None,  # 双流块不量化
            prefix=f"transformer_blocks.{i}",
        )
        for i in range(num_layers)
    ]
)

# 单流块可以量化
self.single_transformer_blocks = nn.ModuleList(
    [
        FluxSingleTransformerBlock(
            dim=self.inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            quant_config=quant_config,  # 单流块进行量化
            prefix=f"single_transformer_blocks.{i}",
        )
        for i in range(num_single_layers)
    ]
)
```

---

## 3. 量化架构设计

### 3.1 支持的量化方法

根据 vllm-omni 现有能力，支持以下量化方法:

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| FP8 | 8-bit 浮点量化 | GPU (H100, H20, A100) |
| INT8 | 8-bit 整数量化 | GPU + NPU |
| GGUF | GGUF 格式量化 | CPU 推理 |
| INC/AutoRound | Intel Neural Compressor | 离线量化 |

### 3.2 需要量化的组件

```
OvisImageTransformer2DModel
│
├── x_embedder: Linear              ✓ 可量化
├── context_embedder: Linear        ✓ 可量化
│
├── transformer_blocks (×6)
│   ├── norm1.linear                ✗ 保持全精度 (调制层)
│   ├── norm1_context.linear        ✗ 保持全精度 (调制层)
│   ├── attn.to_qkv                 ✓ 可量化
│   ├── attn.add_kv_proj            ✓ 可量化
│   ├── attn.to_out[0]              ✓ 可量化
│   ├── attn.to_add_out             ✓ 可量化
│   ├── ff                          ✓ 可量化
│   └── ff_context                  ✓ 可量化
│
├── single_transformer_blocks (×27)
│   ├── norm.linear                 ✗ 保持全精度 (调制层)
│   ├── proj_mlp                    ✓ 可量化
│   ├── proj_out                    ✓ 可量化
│   └── attn.to_qkv                 ✓ 可量化
│
├── norm_out.linear                 ✗ 保持全精度 (最终调制)
└── proj_out: Linear                ✓ 可量化
```

### 3.3 不应量化的组件

根据 Flux 的经验和 Issue #2728 的讨论:

1. **AdaLayerNorm 调制层** (`norm1.linear`, `norm1_context.linear`, `norm.linear`):
   - 原因: 调制层输出的 shift/scale/gate 直接乘入残差流
   - 影响: 量化会引入累积误差，严重影响生成质量

2. **最终输出调制层** (`norm_out`):
   - 原因: 最终调制直接影响输出投影
   - 影响: FP8 量化会导致输出噪声

3. **位置编码层**: 无需量化，计算量小

---

## 4. 详细修改方案

### 4.1 OvisImageAttention 修改

**文件**: `vllm_omni/diffusion/models/ovis_image/ovis_image_transformer.py`

```python
class OvisImageAttention(nn.Module):
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
        quant_config: "QuantizationConfig | None" = None,  # 新增
        prefix: str = "",  # 新增
    ):
        super().__init__()
        
        # ... 其他初始化代码 ...
        
        # 修改: 添加 quant_config 和 prefix
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
                RowParallelLinear(  # 改为 RowParallelLinear 以支持量化
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
            # ... norm 层保持不变 ...
            
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
```

**关键修改点**:
1. 添加 `quant_config` 和 `prefix` 参数
2. 移除 `disable_tp=True`（与量化不兼容）
3. 将 `to_out[0]` 改为 `RowParallelLinear`
4. 将 `to_add_out` 改为 `RowParallelLinear`
5. 为所有 Linear 层添加 `prefix`

### 4.2 OvisImageTransformerBlock 修改

```python
class OvisImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: "QuantizationConfig | None" = None,  # 新增
        prefix: str = "",  # 新增
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
        self.ff = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="swiglu",
            quant_config=quant_config,
            prefix=f"{prefix}.ff",
        )
        
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="swiglu",
            quant_config=quant_config,
            prefix=f"{prefix}.ff_context",
        )
```

### 4.3 OvisImageSingleTransformerBlock 修改

```python
class OvisImageSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: "QuantizationConfig | None" = None,  # 新增
        prefix: str = "",  # 新增
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
```

### 4.4 OvisImageTransformer2DModel 修改

```python
class OvisImageTransformer2DModel(nn.Module):
    # ... 文档字符串 ...
    
    _repeated_blocks = ["OvisImageTransformerBlock", "OvisImageSingleTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]
    
    # 添加 packed_modules_mapping 支持权重加载
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
        quant_config: "QuantizationConfig | None" = None,  # 新增
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
        # 如果内存受限，可以选择性量化
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
```

### 4.5 OvisImagePipeline 修改

```python
# pipeline_ovis_image.py

class OvisImagePipeline(nn.Module, CFGParallelMixin, DiffusionPipelineProfilerMixin):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        # ... 其他初始化代码 ...
        
        # 传递 quant_config 到 transformer
        self.transformer = OvisImageTransformer2DModel(
            od_config=od_config,
            quant_config=od_config.quantization_config,  # 新增
        )
```

### 4.6 FeedForward 层适配

需要从 diffusers 的 FeedForward 改为支持量化的版本:

```python
# ovis_image_transformer.py

from vllm_omni.diffusion.models.flux.flux_transformer import FeedForward

# 或自定义实现
class OvisImageFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        activation_fn: str = "swiglu",
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        
        # SwiGLU: gate * up_proj
        self.gate_proj = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            inner_dim,
            dim_out,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

---

## 5. 量化策略建议

### 5.1 推荐配置

根据 Flux 的经验，推荐以下配置:

**配置 A: 平衡质量与显存 (推荐)**
```yaml
quantization:
  method: fp8
  # 仅量化单流块
  transformer_blocks: null  # 不量化
  single_transformer_blocks: fp8
```

**配置 B: 最大显存节省**
```yaml
quantization:
  method: int8
  # 量化所有可量化层
```

**配置 C: 选择性量化**
```yaml
quantization:
  method: fp8
  # 通过 ignored_layers 排除敏感层
  ignored_layers:
    - "transformer_blocks.*"
    - "norm_out.*"
    - "*.norm.*"
```

### 5.2 量化后显存对比

| 配置 | 模型大小 (FP16) | 量化后大小 | 节省比例 |
|------|----------------|-----------|----------|
| 仅单流块 FP8 | ~14 GB | ~8 GB | ~43% |
| 全部 FP8 | ~14 GB | ~7 GB | ~50% |
| 全部 INT8 | ~14 GB | ~7 GB | ~50% |

---

## 6. 前向传播兼容性处理

### 6.1 FP8 量化需要 contiguous 输入

```python
# ovis_image_transformer.py - OvisImageAttention.forward

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
    
    # ... 注意力计算 ...
    
    if encoder_hidden_states is not None:
        encoder_hidden_states = encoder_hidden_states.contiguous()
        # ... 编码器注意力计算 ...
        
        # FP8 RowParallelLinear 需要 contiguous 输入
        hidden_states = self.to_out[0](hidden_states.contiguous())
        encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())
        
        return hidden_states, encoder_hidden_states
```

### 6.2 处理 TP 相关的张量操作

```python
# 在需要 AllGather 的地方添加处理
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather

def forward(...):
    # ...
    if encoder_hidden_states is None and get_tensor_model_parallel_world_size() > 1:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=-1)
    return hidden_states
```

---

## 7. 权重加载兼容性

### 7.1 支持量化权重加载

```python
# load_weights 方法需要支持量化参数

def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        (".to_qkv", ".to_q", "q"),
        (".to_qkv", ".to_k", "k"),
        (".to_qkv", ".to_v", "v"),
        (".add_kv_proj", ".add_q_proj", "q"),
        (".add_kv_proj", ".add_k_proj", "k"),
        (".add_kv_proj", ".add_v_proj", "v"),
    ]
    
    params_dict = dict(self.named_parameters())
    
    # 加载量化相关参数 (weight_scale, etc.)
    for name, buffer in self.named_buffers():
        if name.endswith(".weight_scale") or name.endswith(".scale"):
            params_dict[name] = buffer
    
    # ... 其余加载逻辑 ...
```

---

## 8. 测试验证

### 8.1 单元测试

```python
def test_quantization_support():
    """测试量化配置正确传递"""
    config = OmniDiffusionConfig(
        model="test_model",
        quantization_config="fp8",
    )
    model = OvisImageTransformer2DModel(od_config=config, quant_config=config.quantization_config)
    
    # 验证量化层
    for block in model.single_transformer_blocks:
        assert hasattr(block.attn.to_qkv, 'quant_config')
    
    # 验证非量化层
    for block in model.transformer_blocks:
        assert block.attn.to_qkv.quant_config is None
```

### 8.2 质量验证

```python
def test_generation_quality():
    """比较量化前后生成质量"""
    # 1. 加载 FP16 模型生成参考图像
    # 2. 加载量化模型生成测试图像
    # 3. 比较 SSIM, PSNR 指标
    # 4. 确保 FID 差异在可接受范围内
```

---

## 9. 完整修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `ovis_image_transformer.py` | 主要修改 | 添加量化支持 |
| `pipeline_ovis_image.py` | 小修改 | 传递 quant_config |
| `adalayernorm.py` | 无需修改 | 已支持 quant_config |
| `__init__.py` | 无需修改 | 已导出正确类 |

---

## 10. 设计理由总结

### 10.1 为什么双流块不建议量化

1. **信息流敏感性**: 双流块中，文本和图像信息通过联合注意力进行深度融合，量化噪声会在两个模态间传播放大

2. **Flux 经验**: 根据 Flux 的 Issue #2728，双流块使用 FP8 会导致输出噪声，影响生成质量

3. **层数相对较少**: OvisImage 只有 6 层双流块，相比 27 层单流块，量化收益有限

### 10.2 为什么调制层保持全精度

1. **调制机制**: AdaLayerNorm 输出的 shift/scale/gate 直接乘入残差流，任何量化误差都会被放大

2. **参数量小**: 调制层参数量相对较小，量化收益有限

3. **训练时不稳定性**: 量化调制层可能导致推理时数值溢出或下溢

### 10.3 为什么使用 prefix 参数

1. **权重加载**: prefix 用于正确映射预训练权重

2. **选择性量化**: 可通过 prefix 匹配特定层进行选择性量化

3. **调试友好**: 便于追踪量化参数对应的具体层

### 10.4 为什么移除 disable_tp

1. **量化兼容性**: Tensor Parallel 与量化需要协同工作

2. **性能优化**: TP 可以减少单卡显存占用，与量化配合效果更好

3. **代码一致性**: 与 Flux 等模型保持一致的实现方式
