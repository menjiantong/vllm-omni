# OvisImage 注意力层解析

## OvisImageAttention 的三个投影层

### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OvisImageAttention (Joint Attention)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  hidden_states (图像特征)                                            │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────┐        ┌──────────────────────────────┐            │
│  │  to_qkv     │───────►│ Q_img, K_img, V_img          │            │
│  │ (QKVParallel)│       └──────────────────────────────┘            │
│  └─────────────┘                    │                                │
│                                      │                               │
│  encoder_hidden_states (文本特征)    │  Concat along sequence dim     │
│       │                              ▼                               │
│  ┌─────────────┐        ┌──────────────────────────────┐            │
│  │ add_kv_proj │───────►│ Q_txt, K_txt, V_txt          │            │
│  │ (QKVParallel)│       └──────────────────────────────┘            │
│  └─────────────┘                    │                                │
│                                      ▼                               │
│                          ┌─────────────────────┐                     │
│                          │  Attention 计算      │                     │
│                          │  [Q_txt+Q_img]      │                     │
│                          │  [K_txt+K_img]      │                     │
│                          │  [V_txt+V_img]      │                     │
│                          └─────────────────────┘                     │
│                                      │                               │
│                       ┌──────────────┴──────────────┐               │
│                       ▼                              ▼               │
│              encoder_attn_out                image_attn_out          │
│                       │                              │               │
│                ┌──────┴──────┐               ┌──────┴──────┐        │
│                │ to_add_out  │               │   to_out    │        │
│                │(RowParallel)│               │ (RowParallel)│        │
│                └──────┬──────┘               └──────┬──────┘        │
│                       ▼                              ▼               │
│               txt_output                    img_output               │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

| 层名称 | 作用 | 输入 | 输出 |
|--------|------|------|------|
| `to_qkv` | 图像特征的 Q/K/V 投影 | `hidden_states` (图像) | Q_img, K_img, V_img |
| `add_kv_proj` | 文本条件的 Q/K/V 投影 | `encoder_hidden_states` (文本) | Q_txt, K_txt, V_txt |
| `to_add_out` | 文本注意力输出投影 | encoder_attn_out | txt_output (回到文本维度) |

### 核心逻辑（代码片段）

```python
# 1. to_qkv: 图像特征投影
qkv, _ = self.to_qkv(hidden_states)
query, key, value = qkv.chunk(3, dim=-1)

# 2. add_kv_proj: 文本特征投影（条件存在时）
if self.added_kv_proj_dim is not None:
    encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
    encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)

    # 3. 拼接：文本和图像一起做注意力
    query = torch.cat([encoder_query, query], dim=1)  # [txt_len + img_len, heads, head_dim]
    key = torch.cat([encoder_key, key], dim=1)
    value = torch.cat([encoder_value, value], dim=1)

# 4. Attention 计算
hidden_states = self.attn(query, key, value)

# 5. 分离输出并分别投影
encoder_hidden_states, hidden_states = hidden_states[:, :enc_len], hidden_states[:, enc_len:]
hidden_states, _ = self.to_out[0](hidden_states)           # 图像输出
encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states)  # 文本输出
```

### 为什么叫 "Joint Attention"？

这是一种**双流 DiT** 架构（类似 Flux、SD3）：
- **图像 token** 和 **文本 token** 拼接在一起做自注意力
- 让图像和文本深度交互，互相"看到"对方
- 比传统的 Cross-Attention（图像只看文本）更强大

### 与传统 Cross-Attention 的区别

| 方式 | Q 来源 | K/V 来源 | 特点 |
|------|--------|----------|------|
| Cross-Attention | 图像 | 文本 | 图像单向查询文本 |
| Joint Attention | 图像+文本 | 图像+文本 | 双向交互，文本也会更新 |

这种设计在 OvisImage、Flux、SD3 等现代 DiT 模型中很常见。

---

## Tensor Parallel 线性层选择

### 四种线性层

| 层名称 | 切分方式 | 使用场景 |
|--------|----------|----------|
| `ColumnParallelLinear` | 切分输出维度 | FFN 的 gate_proj、up_proj |
| `RowParallelLinear` | 切分输入维度 | FFN 的 down_proj，Attention 的 o_proj |
| `QKVParallelLinear` | 按注意力头切分 | Attention 的 Q/K/V 投影 |
| `ReplicatedLinear` | 不切分 | 小参数层、Embedding |

### OvisImageAttention 的 TP 改造

```python
# 图像 Q/K/V 投影 - 按头切分
self.to_qkv = QKVParallelLinear(
    hidden_size=query_dim,
    head_size=self.head_dim,
    total_num_heads=self.heads,
    bias=bias,
)

# 文本 Q/K/V 投影 - 按头切分
self.add_kv_proj = QKVParallelLinear(
    hidden_size=self.added_kv_proj_dim,
    head_size=self.head_dim,
    total_num_heads=self.heads,
    bias=added_proj_bias,
)

# 图像输出投影 - 行并行（聚合）
self.to_out = RowParallelLinear(
    self.inner_dim, self.out_dim,
    bias=out_bias,
    input_is_parallel=True
)

# 文本输出投影 - 行并行（聚合）
self.to_add_out = RowParallelLinear(
    input_size=self.inner_dim,
    output_size=query_dim,
    bias=out_bias,
    input_is_parallel=True
)
```

### 通信优化原理

```
ColumnParallel ──→ 计算 ──→ RowParallel
      │                           │
      │    中间无需通信            │
      └───────────────────────────┘
                                  │
                                  ▼
                           All-Reduce（仅一次）
```

这种 `ColumnParallel → RowParallel` 组合最小化了通信次数，是最优的 TP 模式。
