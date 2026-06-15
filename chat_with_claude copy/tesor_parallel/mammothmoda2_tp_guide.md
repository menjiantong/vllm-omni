# MammothModa2 Tensor Parallel 实现说明

## 概述

本实现为 MammothModa2 模型添加了完整的 Tensor Parallel 支持，遵循 vLLM Omni 的 TP 实现规范。

## 文件结构

```
vllm_omni/diffusion/models/mammoth_moda2/
├── mammothmoda2_dit_model_tp.py   # TP 版本的 Transformer 模型
├── pipeline_mammothmoda2_dit_tp.py # TP 版本的 Pipeline
├── weight_loader.py               # TP 权重加载工具
├── mammothmoda2_dit_model.py      # 原始模型（保留）
└── pipeline_mammothmoda2_dit.py   # 原始 Pipeline（保留）
```

## 主要修改

### 1. FeedForward 层 (SwiGLU)

**原始实现**:
```python
class LuminaFeedForward(nn.Module):
    def __init__(self, dim, inner_dim):
        self.linear_1 = nn.Linear(dim, inner_dim)  # gate
        self.linear_3 = nn.Linear(dim, inner_dim)  # up
        self.linear_2 = nn.Linear(inner_dim, dim)  # down
```

**TP 实现**:
```python
class MammothModa2FeedForward(nn.Module):
    def __init__(self, dim, inner_dim):
        # MergedColumnParallelLinear: 合并 gate + up
        self.w13 = MergedColumnParallelLinear(dim, [inner_dim, inner_dim])
        self.act = SiluAndMul()  # gate * up
        # RowParallelLinear: down with all-reduce
        self.w2 = RowParallelLinear(inner_dim, dim, input_is_parallel=True)
```

**权重映射**:
```
linear_1.weight + linear_3.weight -> w13.weight (列切分)
linear_2.weight                   -> w2.weight  (行切分)
```

### 2. Attention 层

**原始实现**:
```python
# 使用 diffusers 的 Attention
self.attn = Attention(query_dim=dim, heads=num_heads, ...)
# 内部有 to_q, to_k, to_v, to_out
```

**TP 实现**:
```python
class MammothModa2Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads):
        # QKVParallelLinear: 合并 Q, K, V 投影
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
        )
        # RowParallelLinear: output projection
        self.to_out = RowParallelLinear(dim, dim, input_is_parallel=True)
```

**权重映射**:
```
to_q.weight + to_k.weight + to_v.weight -> to_qkv.weight (列切分)
to_out.0.weight                         -> to_out.0.weight (行切分)
```

### 3. TP 约束验证

```python
def validate_mammothmoda2_tp_constraints(
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    ffn_inner_dim: int,
    tensor_parallel_size: int,
) -> list[int]:
    """验证 TP 约束，返回支持的 TP 配置列表。"""
```

**约束条件**:
- `dim % tp_size == 0`
- `num_heads % tp_size == 0`
- `num_kv_heads % tp_size == 0`
- `ffn_inner_dim % tp_size == 0`

### 4. HSDP 支持

模型同时支持 HSDP (Hybrid Sharded Data Parallel):

```python
@staticmethod
def _is_transformer_block(name: str, module) -> bool:
    return (
        "layers" in name
        or "noise_refiner" in name
        or "context_refiner" in name
        or "ref_image_refiner" in name
    ) and name.split(".")[-1].isdigit()

_hsdp_shard_conditions = [_is_transformer_block]
```

## 使用方式

### Python API

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# 配置 TP
parallel_config = DiffusionParallelConfig(
    tensor_parallel_size=2,  # 使用 2 GPU
)

# 加载模型
omni = Omni(
    model="your-model-path",
    parallel_config=parallel_config,
)

# 生成
output = omni.generate(
    "a beautiful landscape",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

### 命令行

```bash
python examples/offline_inference/mammothmodal2_preview/run_mammothmoda2_t2i.py \
    --model your-model-path \
    --prompt "a beautiful landscape" \
    --tensor-parallel-size 2 \
    --output "output.png"
```

## 权重加载

TP 模型需要特殊的权重加载逻辑来处理权重映射和切分：

```python
from vllm_omni.diffusion.models.mammoth_moda2.weight_loader import load_mammothmoda2_weights_tp

# 加载权重时会自动：
# 1. 合并 gate + up 权重并切分 -> w13
# 2. 合并 Q + K + V 权重并切分 -> to_qkv
# 3. 切分 down 和 output 权重 (行切分)
```

## 性能预期

| 配置 | 显存减少 | 理论加速 |
|------|---------|---------|
| TP=2 | ~50% | 1.8-1.9x |
| TP=4 | ~75% | 3.5-3.8x |

## 与原始模型的关系

- **原始模型** (`mammothmoda2_dit_model.py`): 保持不变，用于单 GPU 推理
- **TP 模型** (`mammothmoda2_dit_model_tp.py`): 新增，用于多 GPU 推理

Pipeline 会根据 `tensor_parallel_size` 自动选择使用哪个版本：
- `tp_size=1`: 使用原始模型
- `tp_size>1`: 使用 TP 模型

## 测试

```bash
# 运行 TP 测试
pytest tests/diffusion/models/mammoth_moda2/test_mammothmoda2_tp.py -v
```

## 注意事项

1. **维度约束**: 确保 `num_heads`, `num_kv_heads`, `hidden_size`, `ffn_inner_dim` 能被 `tp_size` 整除
2. **权重格式**: 需要使用原始 HuggingFace 格式的权重（非 TP 切分格式）
3. **混合精度**: 支持 FP16/BF16，但调制层保持 FP32 以保证精度

## 参考

- [Tensor Parallel 文档](/docs/design/feature/tensor_parallel.md)
- [Z-Image TP 实现](/vllm_omni/diffusion/models/z_image/z_image_transformer.py)
- [FLUX TP 实现](/vllm_omni/diffusion/models/flux/flux_transformer.py)
