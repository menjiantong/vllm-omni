# Tensor Parallel 线性层详解

## 四种线性层概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Tensor Parallel 线性层                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐                                            │
│  │ ReplicatedLinear    │  每个 GPU 持有完整副本，不切分             │
│  └─────────────────────┘                                            │
│                                                                      │
│  ┌─────────────────────┐                                            │
│  │ ColumnParallelLinear│  切分输出维度 (Output dimension)           │
│  └─────────────────────┘                                            │
│                                                                      │
│  ┌─────────────────────┐                                            │
│  │ RowParallelLinear   │  切分输入维度 (Input dimension)            │
│  └─────────────────────┘                                            │
│                                                                      │
│  ┌─────────────────────┐                                            │
│  │ QKVParallelLinear   │  按注意力头切分 (Attention heads)          │
│  └─────────────────────┘                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. ReplicatedLinear — 复制层

### 原理
每个 GPU 持有权重的完整副本，不做任何切分。

### 使用场景
- 参数量较小的层
- Embedding 层
- 最终的 LM Head（有时）
- 需要全局同步的层

### 参数说明

```python
ReplicatedLinear(
    input_size: int,          # 输入维度
    output_size: int,         # 输出维度
    bias: bool = True,        # 是否添加 bias
    skip_bias_add: bool = False,  # 跳过 bias 加法，返回 bias
    params_dtype: torch.dtype | None = None,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",         # 参数名前缀（用于加载权重）
    return_bias: bool = True, # 是否在 forward 中返回 bias
    disable_tp: bool = False, # 禁用 TP（对 ReplicatedLinear 无效）
)
```

### 数据流

```
输入 X [batch, input_dim]  (每个 GPU 相同)
         │
         ▼
    ┌─────────────┐
    │   Weight    │  [output_dim, input_dim] (每个 GPU 完整)
    └─────────────┘
         │
         ▼
输出 Y [batch, output_dim] (每个 GPU 相同)
```

---

## 2. ColumnParallelLinear — 列并行

### 原理
切分**输出维度**，每个 GPU 持有输出的一部分。

### 使用场景
- FFN 的 `gate_proj`、`up_proj`
- 任何需要扩展维度的层
- 作为 `ColumnParallel → RowParallel` 组合的前半部分

### 参数说明

```python
ColumnParallelLinear(
    input_size: int,              # 输入维度
    output_size: int,             # 输出维度（会被切分）
    bias: bool = True,
    gather_output: bool = False,  # ⭐ 关键参数：是否 All-Gather 输出
    skip_bias_add: bool = False,
    params_dtype: torch.dtype | None = None,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    return_bias: bool = True,
    disable_tp: bool = False,
)
```

### `gather_output` 参数详解

| gather_output | 行为 | 通信 | 使用场景 |
|---------------|------|------|----------|
| `False` (默认) | 输出保持切分状态 | 无通信 | 后接 RowParallelLinear |
| `True` | All-Gather 聚合输出 | All-Gather | 需要完整输出的场景 |

### 数据流

```
输入 X [batch, input_dim]  (每个 GPU 相同)
         │
         ▼
    ┌─────────────┐
    │  Weight_i   │  [output_dim/tp, input_dim] (每个 GPU 不同)
    └─────────────┘
         │
         ▼
输出 Y_i [batch, output_dim/tp] (每个 GPU 不同部分)

如果 gather_output=True:
         │
         ▼ All-Gather
输出 Y [batch, output_dim] (每个 GPU 完整)
```

### 示例

```python
# FFN 的 gate_proj 和 up_proj
self.gate_proj = ColumnParallelLinear(
    hidden_size, intermediate_size,
    bias=False,
    gather_output=False,  # 不聚合，后续接 RowParallel
)

self.up_proj = ColumnParallelLinear(
    hidden_size, intermediate_size,
    bias=False,
    gather_output=False,
)
```

---

## 3. RowParallelLinear — 行并行

### 原理
切分**输入维度**，输出需要 All-Reduce 求和。

### 使用场景
- FFN 的 `down_proj`
- Attention 的 `o_proj`
- 作为 `ColumnParallel → RowParallel` 组合的后半部分

### 参数说明

```python
RowParallelLinear(
    input_size: int,                  # 输入维度（会被切分）
    output_size: int,                 # 输出维度
    bias: bool = True,
    input_is_parallel: bool = True,   # ⭐ 关键参数：输入是否已切分
    skip_bias_add: bool = False,
    params_dtype: torch.dtype | None = None,
    reduce_results: bool = True,      # ⭐ 是否 All-Reduce 输出
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    return_bias: bool = True,
    disable_tp: bool = False,
)
```

### `input_is_parallel` 参数详解 ⭐

这是最重要的参数，决定了输入数据的处理方式。

| input_is_parallel | 输入状态 | 内部操作 | 使用场景 |
|-------------------|----------|----------|----------|
| `True` (默认) | 输入已经按 TP 切分 | 直接计算 | 前一层是 ColumnParallel |
| `False` | 输入是完整的 | 先切分再计算 | 前一层不是 TP 层 |

### `reduce_results` 参数详解

| reduce_results | 行为 | 通信 | 使用场景 |
|----------------|------|------|----------|
| `True` (默认) | All-Reduce 求和输出 | All-Reduce | 正常 TP 场景 |
| `False` | 输出保持切分状态 | 无通信 | 特殊优化场景 |

### 数据流

```
情况1: input_is_parallel=True (默认，接 ColumnParallel)

输入 X_i [batch, input_dim/tp] (每个 GPU 不同部分，来自前一层)
         │
         ▼
    ┌─────────────┐
    │  Weight_i   │  [output_dim, input_dim/tp] (每个 GPU 不同)
    └─────────────┘
         │
         ▼
输出 Y_i [batch, output_dim] (部分结果)
         │
         ▼ All-Reduce (如果 reduce_results=True)
输出 Y [batch, output_dim] (完整结果)


情况2: input_is_parallel=False (输入未切分)

输入 X [batch, input_dim] (每个 GPU 完整)
         │
         ▼ split_tensor_along_last_dim
输入 X_i [batch, input_dim/tp] (每个 GPU 取一部分)
         │
         ▼
    ┌─────────────┐
    │  Weight_i   │
    └─────────────┘
         │
         ▼ All-Reduce
输出 Y [batch, output_dim]
```

### 示例

```python
# FFN 的 down_proj
self.down_proj = RowParallelLinear(
    intermediate_size, hidden_size,
    bias=False,
    input_is_parallel=True,   # 前一层是 ColumnParallel，输入已切分
    reduce_results=True,      # 需要 All-Reduce 聚合结果
)

# Attention 的 o_proj
self.o_proj = RowParallelLinear(
    num_heads * head_dim, hidden_size,
    bias=False,
    input_is_parallel=True,
)
```

---

## 4. QKVParallelLinear — QKV 专用并行

### 原理
专门用于 Attention 的 Q、K、V 投影，按**注意力头**切分。继承自 `ColumnParallelLinear`。

### 使用场景
- Self-Attention 的 Q/K/V 投影
- 支持 GQA (Grouped Query Attention)
- 支持 MQA (Multi-Query Attention)

### 参数说明

```python
QKVParallelLinear(
    hidden_size: int,              # 输入隐藏维度
    head_size: int,                # 每个注意力头的维度
    total_num_heads: int,          # Q 的总注意力头数
    total_num_kv_heads: int | None = None,  # K/V 的总注意力头数（GQA）
    bias: bool = True,
    skip_bias_add: bool = False,
    params_dtype: torch.dtype | None = None,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    return_bias: bool = True,
    disable_tp: bool = False,
    v_head_size: int | None = None,  # V 头的维度（可以与 Q/K 不同）
)
```

### GQA/MQA 支持

```python
# 标准多头注意力 (MHA)
QKVParallelLinear(
    hidden_size=4096,
    head_size=128,
    total_num_heads=32,      # Q: 32 heads
    total_num_kv_heads=32,   # K/V: 32 heads (与 Q 相同)
)

# Grouped Query Attention (GQA)
QKVParallelLinear(
    hidden_size=4096,
    head_size=128,
    total_num_heads=32,      # Q: 32 heads
    total_num_kv_heads=8,    # K/V: 8 heads (Q 的 1/4)
)

# Multi-Query Attention (MQA)
QKVParallelLinear(
    hidden_size=4096,
    head_size=128,
    total_num_heads=32,      # Q: 32 heads
    total_num_kv_heads=1,    # K/V: 1 head (共享)
)
```

### 数据流

```
输入 X [batch, seq_len, hidden_dim]
         │
         ▼
    ┌─────────────────────────────────────────┐
    │           QKVParallelLinear              │
    │  输出按头切分：                           │
    │  - Q: num_heads/tp 个头                  │
    │  - K: num_kv_heads/tp 个头               │
    │  - V: num_kv_heads/tp 个头               │
    └─────────────────────────────────────────┘
         │
         ▼
输出 QKV [batch, seq_len, (Q+K+V) * local_heads * head_dim]
```

### 重要属性

```python
# 每个 GPU 上的头数
qkv_linear.num_heads       # Q 的头数 / tp_size
qkv_linear.num_kv_heads    # K/V 的头数 / tp_size

# K/V 头的复制数（用于 GQA/MQA）
qkv_linear.num_kv_head_replicas  # tp_size / total_num_kv_heads
```

---

## 最佳实践：ColumnParallel → RowParallel 组合

### 为什么这样组合？

```
ColumnParallelLinear          RowParallelLinear
       │                              │
       │ 无需通信                      │ All-Reduce
       └──────────────────────────────┘
                 最优通信模式
```

### 完整示例：FFN 层

```python
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        
        # ColumnParallel: 扩展维度，无通信
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size,
            bias=False,
            gather_output=False,  # 不聚合，保持切分
        )
        
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size,
            bias=False,
            gather_output=False,
        )
        
        # RowParallel: 收缩维度，All-Reduce
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            bias=False,
            input_is_parallel=True,   # 输入已切分
            reduce_results=True,      # All-Reduce 聚合
        )
    
    def forward(self, x):
        # x: [batch, hidden_size] (每个 GPU 完整)
        
        gate, _ = self.gate_proj(x)  # [batch, intermediate/tp]
        up, _ = self.up_proj(x)      # [batch, intermediate/tp]
        
        hidden = gate * F.silu(up)   # 无通信，本地计算
        
        out, _ = self.down_proj(hidden)  # All-Reduce
        # out: [batch, hidden_size]
        
        return out
```

### 完整示例：Attention 层

```python
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        
        # QKV 投影：按头切分
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            bias=False,
        )
        
        # 输出投影：行并行聚合
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            bias=False,
            input_is_parallel=True,
        )
    
    def forward(self, x):
        # x: [batch, seq, hidden]
        
        qkv, _ = self.qkv_proj(x)  # [batch, seq, 3 * local_heads * head_dim]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # ... Attention 计算 ...
        
        out, _ = self.o_proj(attn_output)  # All-Reduce
        return out
```

---

## 通信总结

| 层类型 | 输入处理 | 输出处理 | 通信操作 |
|--------|----------|----------|----------|
| ReplicatedLinear | 完整 | 完整 | 无 |
| ColumnParallelLinear | 完整 | 切分 | 无 (或 All-Gather) |
| RowParallelLinear | 切分/完整 | 完整 | All-Reduce |
| QKVParallelLinear | 完整 | 切分 | 无 |

### 通信量对比

```
假设: hidden_size = H, tp_size = P

ColumnParallel (gather_output=True):
  All-Gather: O(H/P * P) = O(H)

RowParallel (reduce_results=True):
  All-Reduce: O(H * P) = O(H * P)

ColumnParallel → RowParallel 组合:
  仅一次 All-Reduce: O(H * P)
```

---

## 决策树

```
需要切分吗？
  ├─ 否 ──→ ReplicatedLinear
  │
  └─ 是 ──→ 是 Q/K/V 投影吗？
              │
              ├─ 是 ──→ QKVParallelLinear
              │
              └─ 否 ──→ 是 FFN/MLP 的一部分吗？
                        │
                        ├─ 是 ──→ 是扩展层还是收缩层？
                        │         ├─ 扩展 ──→ ColumnParallelLinear
                        │         └─ 收缩 ──→ RowParallelLinear
                        │
                        └─ 否 ──→ 根据前后层决定
                                  ├─ 前一层是 ColumnParallel → RowParallel
                                  └─ 其他情况 → 具体分析
```

---

## 常见错误

### 1. input_is_parallel 设置错误

```python
# ❌ 错误：前一层的输出是完整的，但设置了 input_is_parallel=True
self.proj1 = ReplicatedLinear(dim, dim)  # 输出完整
self.proj2 = RowParallelLinear(dim, dim, input_is_parallel=True)  # 期望输入已切分

# ✅ 正确
self.proj2 = RowParallelLinear(dim, dim, input_is_parallel=False)  # 先切分再计算
```

### 2. gather_output 和 input_is_parallel 不匹配

```python
# ❌ 错误：ColumnParallel 聚合了输出，但 RowParallel 期望输入已切分
self.proj1 = ColumnParallelLinear(dim, dim, gather_output=True)  # 输出完整
self.proj2 = RowParallelLinear(dim, dim, input_is_parallel=True)  # 期望切分

# ✅ 正确
self.proj1 = ColumnParallelLinear(dim, dim, gather_output=False)  # 输出切分
self.proj2 = RowParallelLinear(dim, dim, input_is_parallel=True)  # 期望切分
```

### 3. GQA 头数不能被 TP 整除

```python
# ❌ 错误：K/V 头数不能被 TP 整除
QKVParallelLinear(
    hidden_size=4096,
    head_size=128,
    total_num_heads=32,
    total_num_kv_heads=7,   # 7 不能被 tp_size=4 整除
)

# ✅ 正确
total_num_kv_heads=8,   # 8 可以被 tp_size=4 整除
```
