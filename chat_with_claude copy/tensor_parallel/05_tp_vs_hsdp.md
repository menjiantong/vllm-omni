# Tensor Parallel vs HSDP 对比分析

## 核心区别总结

| 特性 | Tensor Parallel (TP) | HSDP |
|------|---------------------|------|
| **切分对象** | 计算图 + 权重 | 仅权重 |
| **切分粒度** | 按层维度切分 | 按模块整体切分 |
| **计算方式** | 每个 GPU 计算部分结果 | 每个 GPU 完整计算，权重按需 gather |
| **通信模式** | All-Reduce (每次层输出) | All-Gather (每次层输入前) |
| **能否同时使用** | ❌ 互斥 | ❌ 互斥 |
| **主要目标** | 加速 + 省内存 | 省内存为主 |

---

## 1. 设计理念对比

### Tensor Parallel: 计算并行

```
目标: 加速计算 + 减少内存

思路: 把每一层的计算拆分到多个 GPU

┌─────────────────────────────────────────────────────────┐
│                    单层计算                              │
│                                                         │
│   GPU 0: W_0 (部分权重) → 部分输出                       │
│   GPU 1: W_1 (部分权重) → 部分输出                       │
│   GPU 2: W_2 (部分权重) → 部分输出                       │
│   GPU 3: W_3 (部分权重) → 部分输出                       │
│              │                                          │
│              ▼                                          │
│        All-Reduce 求和                                  │
│              │                                          │
│              ▼                                          │
│         完整输出                                        │
└─────────────────────────────────────────────────────────┘

特点: 每个 GPU 只计算一部分，需要通信合并结果
```

### HSDP: 权重分片

```
目标: 减少内存占用 (让大模型能放进小显存)

思路: 权重分散存储，计算时临时 gather

┌─────────────────────────────────────────────────────────┐
│                    权重存储                              │
│                                                         │
│   GPU 0: 存储 Block 0, 4, 8, ... 的权重                 │
│   GPU 1: 存储 Block 1, 5, 9, ... 的权重                 │
│   GPU 2: 存储 Block 2, 6, 10, ... 的权重                │
│   GPU 3: 存储 Block 3, 7, 11, ... 的权重                │
│                                                         │
└─────────────────────────────────────────────────────────┘

计算 Block 0 时:
┌─────────────────────────────────────────────────────────┐
│   GPU 0: All-Gather Block 0 权重 → 计算                 │
│   GPU 1: 从 GPU 0 接收 Block 0 权重 → 计算              │
│   GPU 2: 从 GPU 0 接收 Block 0 权重 → 计算              │
│   GPU 3: 从 GPU 0 接收 Block 0 权重 → 计算              │
│                                                         │
│   所有 GPU 得到相同结果 (数据并行)                       │
└─────────────────────────────────────────────────────────┘

特点: 权重分散存储，计算时所有 GPU 都要完整计算
```

---

## 2. 权重切分方式对比

### Tensor Parallel: 按维度切分

```python
# 一个 Linear 层的权重
W: [in_dim, out_dim]  # 例如 [4096, 4096]

# TP=4 时，每个 GPU 存储的权重
GPU 0: W[:, 0:1024]      # 第 0-1023 列
GPU 1: W[:, 1024:2048]   # 第 1024-2047 列
GPU 2: W[:, 2048:3072]   # 第 2048-3071 列
GPU 3: W[:, 3072:4096]   # 第 3072-4095 列

# 每层都被切分
# 每个 GPU 都有该层的一部分权重
```

### HSDP: 按模块切分

```python
# 一个 Transformer 有 32 个 Block
Blocks: [Block_0, Block_1, ..., Block_31]

# HSDP shard_size=4 时，每个 GPU 存储的权重
GPU 0: Block_0, Block_4, Block_8, Block_12, ... (8 个 Block)
GPU 1: Block_1, Block_5, Block_9, Block_13, ... (8 个 Block)
GPU 2: Block_2, Block_6, Block_10, Block_14, ... (8 个 Block)
GPU 3: Block_3, Block_7, Block_11, Block_15, ... (8 个 Block)

# 每个 Block 内部的权重是完整的
# 不同 GPU 存储不同的 Block
```

---

## 3. 计算流程对比

### Tensor Parallel 计算流程

```
输入 X: [B, S, D] (复制到所有 GPU)

┌────────────────────────────────────────────┐
│ Linear Layer                               │
│                                            │
│   GPU 0: Y_0 = X @ W_0  → 部分 Y          │
│   GPU 1: Y_1 = X @ W_1  → 部分 Y          │
│   GPU 2: Y_2 = X @ W_2  → 部分 Y          │
│   GPU 3: Y_3 = X @ W_3  → 部分 Y          │
│                                            │
│   All-Reduce Sum → 完整 Y                  │
└────────────────────────────────────────────┘

每个 GPU: 计算 1/4 的输出维度
通信频率: 每层一次 All-Reduce
```

### HSDP 计算流程

```
输入 X: [B, S, D] (复制到所有 GPU)

┌────────────────────────────────────────────┐
│ Block 0                                    │
│                                            │
│   Step 1: All-Gather Block 0 权重          │
│   GPU 0 广播自己的 Block 0 权重给其他 GPU  │
│                                            │
│   Step 2: 所有 GPU 用完整权重计算          │
│   GPU 0: Y = X @ W_Block0  → 完整 Y       │
│   GPU 1: Y = X @ W_Block0  → 完整 Y       │
│   GPU 2: Y = X @ W_Block0  → 完整 Y       │
│   GPU 3: Y = X @ W_Block0  → 完整 Y       │
│                                            │
│   Step 3: 释放 gathered 权重               │
└────────────────────────────────────────────┘

每个 GPU: 计算完整输出
通信频率: 每个 Block 一次 All-Gather
```

---

## 4. 内存占用对比

### Tensor Parallel 内存模型

```
模型参数: 14B (Wan2.2)
TP=4

每个 GPU 内存占用:
├── 权重: 14B / 4 = 3.5B
├── 激活值: 约为单 GPU 的 1/4 (切分状态)
├── 临时缓冲: 较小
└── 总计: ~3.5B + 少量激活

优势: 权重和激活都减少
```

### HSDP 内存模型

```
模型参数: 14B (Wan2.2)
HSDP shard_size=4

每个 GPU 内存占用:
├── 权重存储: 14B / 4 = 3.5B (分散存储)
├── 激活值: 与单 GPU 相同 (完整计算)
├── 临时缓冲: All-Gather 的完整 Block 权重
└── 总计: 3.5B + 完整激活 + 临时权重缓冲

优势: 权重减少，但需要临时缓冲
劣势: 激活值不减，需要额外缓冲
```

---

## 5. 通信开销对比

### Tensor Parallel 通信

```
通信类型: All-Reduce (求和)
通信频率: 每个 TP 层一次
每次通信量: output_dim * sizeof(dtype)

示例 (Block 内):
- Attention: 1 次 All-Reduce (output projection)
- FFN: 1 次 All-Reduce (down projection)

总计: 2 次 All-Reduce per Block

特点:
✓ 通信量相对固定
✓ 可以与计算 overlap
```

### HSDP 通信

```
通信类型: All-Gather (收集权重)
通信频率: 每个 sharded module 一次
每次通信量: module 权重大小

示例 (Block 内):
- Block 权重: All-Gather 一次 (包含 attn + ffn 全部权重)

总计: 1 次 All-Gather per Block

特点:
✓ 通信量取决于模块大小
✓ 必须在计算前完成
✗ 较难与计算 overlap
```

---

## 6. 适用场景对比

### Tensor Parallel 适用场景

```
✓ 推荐使用场景:
  - 有充足 GPU 数量
  - 追求推理速度
  - 模型维度可被 TP size 整除
  - 需要同时减少权重和激活内存

✗ 不推荐场景:
  - num_heads 不能被整除
  - GPU 间通信带宽低
  - 只需要减少权重内存
```

### HSDP 适用场景

```
✓ 推荐使用场景:
  - GPU 显存紧张，只需减少权重内存
  - 模型结构不适合 TP (维度不能整除)
  - 不想修改模型代码
  - 只需要让大模型能跑起来

✗ 不推荐场景:
  - 追求最快推理速度
  - 激活内存也是瓶颈
  - GPU 间通信带宽很低
```

---

## 7. 代码修改对比

### Tensor Parallel: 需要修改模型代码

```python
# 需要将所有 nn.Linear 替换为并行层

# 原始代码
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        self.w1 = nn.Linear(dim, hidden_dim)  # ❌
        self.w2 = nn.Linear(hidden_dim, dim)  # ❌

# TP 版本 (需要修改)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        self.w1 = ColumnParallelLinear(dim, hidden_dim)  # ✅
        self.w2 = RowParallelLinear(hidden_dim, dim, input_is_parallel=True)  # ✅

# 还需要:
# - 替换 Attention 的 QKV 投影
# - 添加维度约束验证
# - 处理 head 数的本地化
```

### HSDP: 无需修改模型核心代码

```python
# 只需添加切分条件函数

class MyTransformerModel(nn.Module):
    # 原有代码完全不变
    def __init__(self, ...):
        self.blocks = nn.ModuleList([...])
    
    # 只需添加这个属性
    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "blocks" in name and name.split(".")[-1].isdigit()
    
    _hsdp_shard_conditions = [_is_transformer_block]

# HSDP 通过 FSDP2 自动处理切分
# 不需要修改任何 Linear 层
```

---

## 8. 性能对比示例

### 场景: Wan2.2 14B 模型

```
硬件: 8x A100 80GB

┌─────────────────┬───────────────┬───────────────┐
│     指标        │  TP=4         │  HSDP=4       │
├─────────────────┼───────────────┼───────────────┤
│ 权重内存        │  3.5B         │  3.5B         │
│ 激活内存        │  减少到 1/4   │  不变         │
│ 总显存占用      │  最低         │  中等         │
│ 推理速度        │  最快         │  较慢         │
│ 代码修改量      │  大           │  小           │
│ 维度约束        │  有           │  无           │
└─────────────────┴───────────────┴───────────────┘
```

---

## 9. 选择建议

### 决策流程

```
开始
  │
  ▼
是否需要最快推理速度?
  │
  ├─ 是 → 使用 Tensor Parallel
  │        (确保维度可整除)
  │
  └─ 否
       │
       ▼
     GPU 显存是否足够放下完整模型权重?
       │
       ├─ 是 → 不需要并行 (或用 Data Parallel)
       │
       └─ 否
            │
            ▼
          是否能修改模型代码?
            │
            ├─ 是 → Tensor Parallel (更好的性能)
            │
            └─ 否 → HSDP (最小改动)
```

### 简单规则

| 条件 | 推荐 |
|------|------|
| 追求速度 + 可整除维度 | **Tensor Parallel** |
| 只需省内存 + 不想改代码 | **HSDP** |
| 激活内存也是瓶颈 | **Tensor Parallel** |
| 维度不能整除 | **HSDP** |

---

## 10. 技术限制对比

### Tensor Parallel 限制

```python
# 必须满足的约束
num_heads % tensor_parallel_size == 0
num_kv_heads % tensor_parallel_size == 0

# 例如:
num_heads = 30, tp_size = 4  # ❌ 30 % 4 = 2 ≠ 0
num_heads = 32, tp_size = 4  # ✅ 32 % 4 = 0

# 不能与 HSDP 同时使用
```

### HSDP 限制

```python
# 不能与 Tensor Parallel 同时使用
# standalone HSDP 需要显式指定 shard_size

parallel_config = DiffusionParallelConfig(
    use_hsdp=True,
    hsdp_shard_size=8,  # 必须指定
)

# 如果与其他并行组合 (如 Sequence Parallel)
# 可能自动计算 shard_size
```

---

## 总结

| 方面 | Tensor Parallel | HSDP |
|------|-----------------|------|
| **核心思想** | 计算并行化 | 权重分片存储 |
| **主要收益** | 速度 + 内存 | 内存 |
| **通信类型** | All-Reduce | All-Gather |
| **代码改动** | 大 (替换所有 Linear) | 小 (添加条件函数) |
| **维度约束** | 有 (必须整除) | 无 |
| **激活内存** | 减少 | 不变 |
| **推荐场景** | 追求性能 | 快速支持大模型 |
