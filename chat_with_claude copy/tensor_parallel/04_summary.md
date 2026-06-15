# Tensor Parallel 快速参考

## 核心概念速查

### 1. 两种切分方式对比

| 特性 | Column Parallel | Row Parallel |
|------|-----------------|--------------|
| 切分维度 | 输出维度 (列) | 输入维度 (行) |
| 输入状态 | 复制 | 切分 |
| 输出状态 | 切分 | 复制 |
| 通信需求 | 无 | All-Reduce |
| 权重形状 | `[in, out/N]` | `[in/N, out]` |
| 典型用途 | FFN up, QKV | FFN down, Attn out |

### 2. 标准配对模式

```
输入 (复制)
    │
    ▼
┌─────────────────────┐
│  ColumnParallel     │  ← 扩展维度
└─────────────────────┘
    │
    ▼
中间激活 (切分, 无通信)
    │
    ▼
┌─────────────────────┐
│   RowParallel       │  ← 收缩维度
└─────────────────────┘
    │
    ▼
All-Reduce (通信)
    │
    ▼
输出 (复制)
```

---

## API 速查

### ColumnParallelLinear

```python
from vllm.model_executor.layers.linear import ColumnParallelLinear

layer = ColumnParallelLinear(
    input_size,        # 输入维度
    output_size,       # 输出维度
    bias=False,        # 是否有偏置
    return_bias=False, # 是否返回单独的 bias
)
```

### RowParallelLinear

```python
from vllm.model_executor.layers.linear import RowParallelLinear

layer = RowParallelLinear(
    input_size,              # 输入维度
    output_size,             # 输出维度
    bias=False,
    input_is_parallel=True,  # ⚠️ 重要: 输入是否已切分
    return_bias=False,
)
```

### QKVParallelLinear

```python
from vllm.model_executor.layers.linear import QKVParallelLinear

layer = QKVParallelLinear(
    hidden_size=dim,              # 输入维度
    head_size=head_dim,           # 每个 head 的维度
    total_num_heads=num_heads,    # 总 head 数
    total_num_kv_heads=num_kv_heads,  # 总 KV head 数
    bias=False,
    return_bias=False,
)

# 访问本地 head 数
local_heads = layer.num_heads        # 总数 / tp_size
local_kv_heads = layer.num_kv_heads  # 总数 / tp_size
```

---

## 约束条件

| 维度 | 约束 | 示例 |
|------|------|------|
| `num_heads` | 能被 tp_size 整除 | 32 heads, tp=4 ✓ (8 heads/GPU) |
| `num_kv_heads` | 能被 tp_size 整除 | 同上 |
| `ff_hidden_dim` | 能被 tp_size 整除 (可选) | 4096, tp=4 ✓ (1024/GPU) |

---

## 常见错误排查

### 错误 1: 维度不匹配

```
RuntimeError: shape mismatch in linear layer
```

**原因**: 使用了总 head 数而不是本地 head 数

```python
# ❌ 错误
q_size = self.num_heads * self.head_dim

# ✅ 正确
q_size = self.to_qkv.num_heads * self.head_dim
```

### 错误 2: TP 没有生效

**症状**: 显存没有减少，速度没有提升

**检查**:
1. 是否还在使用 `nn.Linear`？
2. 是否设置了 `tensor_parallel_size`？
3. 是否设置了 `input_is_parallel=True`？

### 错误 3: 整除约束失败

```
ValueError: num_heads (30) must be divisible by tensor_parallel_size (4)
```

**解决**:
- 修改模型的 `num_heads` 为能被 tp_size 整除的数
- 或使用不同的 tp_size

---

## 快速转换模板

### MLP 转换

```python
# 原始
self.w1 = nn.Linear(dim, hidden_dim, bias=False)
self.w2 = nn.Linear(hidden_dim, dim, bias=False)

# TP 版本
self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, return_bias=False)
self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, 
                             input_is_parallel=True, return_bias=False)
```

### Attention 转换

```python
# 原始
self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
self.to_out = nn.Linear(dim, dim, bias=False)

# TP 版本
self.to_qkv = QKVParallelLinear(
    hidden_size=dim,
    head_size=head_dim,
    total_num_heads=num_heads,
    total_num_kv_heads=num_kv_heads,
    bias=False,
    return_bias=False,
)
self.to_out = RowParallelLinear(dim, dim, bias=False,
                                  input_is_parallel=True, return_bias=False)
```

---

## 性能预期

| GPU 数量 | 显存减少 | 理论加速 | 实际加速* |
|---------|---------|---------|----------|
| 2 | ~50% | 2x | 1.8-1.9x |
| 4 | ~75% | 4x | 3.5-3.8x |
| 8 | ~87.5% | 8x | 7-7.5x |

*实际加速取决于模型大小和通信开销

---

## 相关文档

- [01_principles.md](01_principles.md) - 原理详解
- [02_algorithm.md](02_algorithm.md) - 算法逻辑与调用过程
- [03_implementation_guide.md](03_implementation_guide.md) - 实现指南

## 参考实现

| 模型 | 路径 |
|------|------|
| Z-Image | `vllm_omni/diffusion/models/z_image/z_image_transformer.py` |
| FLUX | `vllm_omni/diffusion/models/flux/flux_transformer.py` |
| Qwen-Image | `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py` |

## 测试文件

| 类型 | 路径 |
|------|------|
| E2E 测试 | `tests/e2e/offline_inference/test_zimage_parallelism.py` |
| 约束测试 | `tests/diffusion/models/z_image/test_zimage_tp_constraints.py` |
