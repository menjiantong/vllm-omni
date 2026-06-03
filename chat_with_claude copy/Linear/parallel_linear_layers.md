# vLLM 并行线性层详解

本文档详细介绍 vLLM 中的五种并行线性层实现，包括其算法机制、参数说明和使用方法。

---

## 目录

1. [概述](#概述)
2. [ReplicatedLinear（复制线性层）](#replicatedlinear)
3. [ColumnParallelLinear（列并行线性层）](#columnparallellinear)
4. [MergedColumnParallelLinear（合并列并行线性层）](#mergedcolumnparallellinear)
5. [QKVParallelLinear（QKV并行线性层）](#qkvparallellinear)
6. [RowParallelLinear（行并行线性层）](#rowparallellinear)
7. [对比总结](#对比总结)

---

## 概述

这些线性层是 vLLM 为**张量并行（Tensor Parallelism, TP）**设计的核心组件。在大模型推理中，单个 GPU 可能无法容纳完整的模型权重，需要将模型切分到多个 GPU 上。这些线性层提供了不同的切分策略：

```
矩阵乘法 Y = XA + b 的并行策略：

列并行：A 按列切分 → A = [A_1, A_2, ..., A_p]
        每个 GPU 计算部分输出，最后 all-gather

行并行：A 按行切分 → A = [A_1; A_2; ...; A_p]^T
        X 按列切分，每个 GPU 计算部分结果，最后 all-reduce
```

---

## ReplicatedLinear

### 功能描述

**复制线性层** - 权重在所有 GPU 上**完全复制**，不做任何切分。

### 算法机制

```
输入 X: [batch, input_size]
权重 A: [input_size, output_size]  (每个GPU持有完整副本)
偏置 b: [output_size]

输出 Y = XA + b
```

每个 GPU 执行完全相同的矩阵乘法，持有完整的权重矩阵。

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_size` | int | 必填 | 输入特征维度 |
| `output_size` | int | 必填 | 输出特征维度 |
| `bias` | bool | True | 是否使用偏置 |
| `skip_bias_add` | bool | False | 是否跳过偏置加法（返回偏置供外部融合） |
| `params_dtype` | torch.dtype | None | 参数数据类型 |
| `quant_config` | QuantizationConfig | None | 量化配置 |
| `prefix` | str | "" | 参数名前缀 |
| `return_bias` | bool | True | 是否在forward中返回偏置 |
| `disable_tp` | bool | False | 对ReplicatedLinear无效果 |

### 使用场景

- 小型线性层（权重不大，复制开销小）
- 需要完整输出的层
- 不适合张量并行的层

### 使用示例

```python
from vllm.model_executor.layers.linear import ReplicatedLinear

# 创建一个简单的复制线性层
layer = ReplicatedLinear(
    input_size=4096,
    output_size=4096,
    bias=True,
)

# 前向传播
x = torch.randn(1, 4096)
output, bias = layer(x)  # output: [1, 4096]
```

---

## ColumnParallelLinear

### 功能描述

**列并行线性层** - 权重矩阵沿**输出维度（列）**切分，每个 GPU 持有部分列。

### 算法机制

```
原始权重矩阵 A: [input_size, output_size]

切分后：
A = [A_1, A_2, ..., A_p]  其中 p = tp_size

每个 GPU_i 持有 A_i: [input_size, output_size/p]

前向计算：
Y_i = X @ A_i           # 每个 GPU 计算部分输出
Y = all_gather([Y_1, Y_2, ..., Y_p])  # 可选：聚合完整输出
```

**数据流图：**
```
         ┌─────────────────────────────────────┐
         │           输入 X (完整)              │
         │         [batch, input_size]          │
         └──────────────┬──────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │  GPU 0  │   │  GPU 1  │   │  GPU 2  │
    │ A[:,0:k]│   │ A[:,k:2k]│   │ A[:,2k:]│
    └────┬────┘   └────┬────┘   └────┬────┘
         │              │              │
         ▼              ▼              ▼
       Y_0            Y_1            Y_2
         │              │              │
         └──────────────┼──────────────┘
                        │
                   all_gather (可选)
                        │
                        ▼
              ┌─────────────────┐
              │   完整输出 Y     │
              └─────────────────┘
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_size` | int | 必填 | 输入特征维度 |
| `output_size` | int | 必填 | 输出特征维度（会被 tp_size 整除） |
| `bias` | bool | True | 是否使用偏置 |
| `gather_output` | bool | False | 是否 all-gather 聚合输出 |
| `skip_bias_add` | bool | False | 是否跳过偏置加法 |
| `params_dtype` | torch.dtype | None | 参数数据类型 |
| `quant_config` | QuantizationConfig | None | 量化配置 |
| `prefix` | str | "" | 参数名前缀 |
| `return_bias` | bool | True | 是否返回偏置 |
| `disable_tp` | bool | False | 禁用张量并行时权重不分片 |

### 使用场景

- Attention 的 Q/K/V 投影层
- MLP 的 gate_proj / up_proj 层
- 需要扩展输出维度的层

### 使用示例

```python
from vllm.model_executor.layers.linear import ColumnParallelLinear

# 创建列并行线性层
layer = ColumnParallelLinear(
    input_size=4096,
    output_size=16384,  # 如果 tp_size=4，每个GPU持有 4096 列
    bias=False,
    gather_output=False,  # 输出保持分片状态
)

# 前向传播
x = torch.randn(1, 4096)
output, bias = layer(x)  # output: [1, 16384/tp_size]
```

---

## MergedColumnParallelLinear

### 功能描述

**合并列并行线性层** - 将多个列并行线性层**合并为一个**，权重沿输出维度拼接。

### 算法机制

```
场景：MLP 中 gate_proj 和 up_proj 经常合并

原始两个矩阵：
  gate_proj: [input_size, intermediate_size]
  up_proj:   [input_size, intermediate_size]

合并后：
  gate_up_proj: [input_size, 2 * intermediate_size]
                = [gate_proj | up_proj]  (沿输出维度拼接)

切分策略：
  每个 GPU_i 持有：
    gate_proj_i: [input_size, intermediate_size/tp_size]
    up_proj_i:   [input_size, intermediate_size/tp_size]

  存储为单个矩阵：[input_size, 2*intermediate_size/tp_size]
```

**数据流图：**
```
权重加载阶段：
┌──────────────────────────────────────────────────┐
│           gate_up_proj (磁盘上可能已融合)          │
│        [input_size, 2*intermediate_size]          │
└───────────────────────┬──────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │  GPU 0  │   │  GPU 1  │   │  GPU 2  │
    │[g0 | u0]│   │[g1 | u1]│   │[g2 | u2]│
    └─────────┘   └─────────┘   └─────────┘

前向计算：
输入 X → 每个 GPU 计算 → 输出分两部分 (gate, up)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_size` | int | 必填 | 输入特征维度 |
| `output_sizes` | List[int] | 必填 | 各分片的输出维度列表，如 `[4096, 4096]` |
| `bias` | bool | True | 是否使用偏置 |
| `gather_output` | bool | False | 是否聚合输出 |
| `skip_bias_add` | bool | False | 是否跳过偏置加法 |
| `params_dtype` | torch.dtype | None | 参数数据类型 |
| `quant_config` | QuantizationConfig | None | 量化配置 |
| `prefix` | str | "" | 参数名前缀 |
| `return_bias` | bool | True | 是否返回偏置 |
| `disable_tp` | bool | False | 禁用时作为复制线性层 |

### 使用场景

- **MLP 层优化**：合并 gate_proj 和 up_proj
- **减少 kernel 调用**：一次矩阵乘法替代两次
- **提高显存利用率**：连续存储提高缓存命中率

### 使用示例

```python
from vllm.model_executor.layers.linear import MergedColumnParallelLinear

# 合并 gate_proj 和 up_proj
layer = MergedColumnParallelLinear(
    input_size=4096,
    output_sizes=[16384, 16384],  # gate_proj 和 up_proj 各 16384
    bias=False,
)

# 前向传播
x = torch.randn(1, 4096)
output, _ = layer(x)  # output: [1, 2*16384/tp_size]

# 后续可以切分输出
gate_output = output[:, :16384//tp_size]
up_output = output[:, 16384//tp_size:]
```

---

## QKVParallelLinear

### 功能描述

**QKV 并行线性层** - 专为 Attention 的 **Q/K/V 投影**设计，支持 GQA（Grouped Query Attention）。

### 算法机制

```
标准 Multi-Head Attention：
  Q, K, V 各有 num_heads 个头

Grouped Query Attention (GQA)：
  Q: num_heads 个头
  K, V: num_kv_heads 个头 (num_kv_heads < num_heads)

切分策略：
  每个 GPU 持有：
    Q: num_heads/tp_size 个头
    K: num_kv_heads/tp_size 个头（若 tp_size > num_kv_heads 则复制）

特殊情况 - K/V 头复制：
  当 tp_size > num_kv_heads 时，K/V 权重需要在多个 GPU 上复制

  例如：tp_size=8, num_heads=32, num_kv_heads=2
    - 每个 GPU 持有 32/8=4 个 Q 头
    - K/V 各有 2 个头，需要每 4 个 GPU 复制同一份 K/V 权重
```

**权重布局：**
```
QKV 融合权重布局（单个 GPU）：
┌─────────────────────────────────────────────────────────┐
│ Q 部分        │ K 部分      │ V 部分      │
│ [num_heads/tp * head_size] │ [num_kv_heads/tp * head_size] │ ...
└─────────────────────────────────────────────────────────┘
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hidden_size` | int | 必填 | 隐藏层维度 |
| `head_size` | int | 必填 | 每个注意力头的维度 |
| `total_num_heads` | int | 必填 | Q 的总头数 |
| `total_num_kv_heads` | int | None | K/V 的总头数，None 时等于 total_num_heads |
| `bias` | bool | True | 是否使用偏置 |
| `skip_bias_add` | bool | False | 是否跳过偏置加法 |
| `params_dtype` | torch.dtype | None | 参数数据类型 |
| `quant_config` | QuantizationConfig | None | 量化配置 |
| `prefix` | str | "" | 参数名前缀 |
| `return_bias` | bool | True | 是否返回偏置 |
| `disable_tp` | bool | False | 禁用张量并行 |
| `v_head_size` | int | None | V 头的维度，None 时等于 head_size |

### 使用场景

- **所有 Transformer 的 Attention 层**
- **支持 GQA/MQA 的模型**：如 LLaMA 2, Mistral, Phi-3
- **减少显存和计算**：K/V 头数小于 Q 头数时节省资源

### 使用示例

```python
from vllm.model_executor.layers.linear import QKVParallelLinear

# 标准 Multi-Head Attention
layer = QKVParallelLinear(
    hidden_size=4096,
    head_size=128,
    total_num_heads=32,  # Q, K, V 各 32 个头
    bias=False,
)

# Grouped Query Attention (GQA)
layer_gqa = QKVParallelLinear(
    hidden_size=4096,
    head_size=128,
    total_num_heads=32,      # Q 有 32 个头
    total_num_kv_heads=8,    # K, V 各 8 个头（共享）
    bias=False,
)

# 前向传播
x = torch.randn(1, 4096)
qkv, _ = layer(x)  # 输出已融合 Q, K, V
```

---

## RowParallelLinear

### 功能描述

**行并行线性层** - 权重矩阵沿**输入维度（行）**切分，每个 GPU 持有部分行。

### 算法机制

```
原始权重矩阵 A: [input_size, output_size]

切分后：
        ┌ A_1 ┐
        │ A_2 │
A =     │ .   │    其中 p = tp_size
        │ .   │
        └ A_p ┘

每个 GPU_i 持有 A_i: [input_size/p, output_size]

前向计算：
X 需要先按列切分：X = [X_1, X_2, ..., X_p]
Y_i = X_i @ A_i           # 每个 GPU 计算部分结果
Y = all_reduce([Y_1, Y_2, ..., Y_p])  # 求和得到完整输出
```

**数据流图：**
```
输入 X: [batch, input_size]
            │
            ▼
    split 按列切分
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
   X_0     X_1     X_2
    │       │       │
    ▼       ▼       ▼
┌───────┐┌───────┐┌───────┐
│GPU 0  ││GPU 1  ││GPU 2  │
│A[0:k,:]││A[k:2k,:]││A[2k:,:]│
└───┬───┘└───┬───┘└───┬───┘
    │       │       │
    ▼       ▼       ▼
   Y_0     Y_1     Y_2
    │       │       │
    └───────┼───────┘
            │
       all_reduce (sum)
            │
            ▼
    ┌───────────────┐
    │ 完整输出 Y     │
    │ + bias (仅rank0)│
    └───────────────┘
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_size` | int | 必填 | 输入特征维度（会被 tp_size 整除） |
| `output_size` | int | 必填 | 输出特征维度 |
| `bias` | bool | True | 是否使用偏置 |
| `input_is_parallel` | bool | True | 输入是否已分片 |
| `skip_bias_add` | bool | False | 是否跳过偏置加法 |
| `params_dtype` | torch.dtype | None | 参数数据类型 |
| `reduce_results` | bool | True | 是否 all-reduce 聚合结果 |
| `quant_config` | QuantizationConfig | None | 量化配置 |
| `prefix` | str | "" | 参数名前缀 |
| `return_bias` | bool | True | 是否返回偏置 |
| `disable_tp` | bool | False | 禁用张量并行 |

### 使用场景

- MLP 的 **down_proj** 层
- Attention 的 **o_proj** 层
- 需要聚合分片结果的层

### 使用示例

```python
from vllm.model_executor.layers.linear import RowParallelLinear

# 创建行并行线性层
layer = RowParallelLinear(
    input_size=16384,   # 如果 tp_size=4，每个GPU处理 4096 维输入
    output_size=4096,
    bias=False,
    input_is_parallel=True,  # 输入已经是分片的
    reduce_results=True,     # 自动 all-reduce
)

# 前向传播（输入应该是之前列并行层的分片输出）
x_parallel = torch.randn(1, 16384//tp_size)
output, bias = layer(x_parallel)  # output: [1, 4096] (完整输出)
```

---

## 对比总结

### 并行策略对比

| 层类型 | 切分维度 | 输出聚合方式 | 典型用途 |
|--------|----------|--------------|----------|
| ReplicatedLinear | 不切分 | 无需聚合 | 小型层、嵌入层 |
| ColumnParallelLinear | 输出维度 | all-gather | Q/K/V、gate/up_proj |
| MergedColumnParallelLinear | 输出维度 | all-gather | gate+up_proj 合并 |
| QKVParallelLinear | 输出维度（按头） | 无需聚合 | Attention QKV |
| RowParallelLinear | 输入维度 | all-reduce | o_proj、down_proj |

### 典型 Transformer 层的并行组合

```
                      ┌─────────────────┐
                      │    输入 hidden   │
                      └────────┬────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │    QKVParallelLinear           │  ← 列并行
              │    (Q, K, V 投影)               │
              └────────────────┬───────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │   Attention     │
                      │   (分片计算)     │
                      └────────┬────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │    RowParallelLinear           │  ← 行并行
              │    (o_proj)                    │
              └────────────────┬───────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │    残差连接      │
                      └────────┬────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │  MergedColumnParallelLinear    │  ← 列并行（合并）
              │    (gate_proj + up_proj)       │
              └────────────────┬───────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │   激活函数       │
                      └────────┬────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │    RowParallelLinear           │  ← 行并行
              │    (down_proj)                 │
              └────────────────┬───────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │    残差连接      │
                      └─────────────────┘
```

### 选择指南

| 场景 | 推荐层类型 |
|------|-----------|
| Attention Q/K/V 投影 | `QKVParallelLinear` |
| Attention 输出投影 (o_proj) | `RowParallelLinear` |
| MLP gate_proj / up_proj | `MergedColumnParallelLinear` |
| MLP down_proj | `RowParallelLinear` |
| 小型嵌入层 | `ReplicatedLinear` |
| 单独的扩展层 | `ColumnParallelLinear` |

---

## 参考代码位置

- 源码路径：`vllm/model_executor/layers/linear.py`
- LoRA 扩展：
  - `vllm/lora/layers/column_parallel_linear.py`
  - `vllm/lora/layers/row_parallel_linear.py`
  - `vllm/lora/layers/replicated_linear.py`
