# Tensor Parallel (TP) 原理解析

## 1. 什么是 Tensor Parallel？

Tensor Parallel (TP) 是一种**模型并行**技术，它将模型权重**切分**到多个 GPU 上。每个 GPU 只保存模型参数的一部分，并计算每一层输出的部分结果。

### 核心思想

```
单 GPU:                    多 GPU (TP=2):
┌─────────────┐           ┌───────┐  ┌───────┐
│   Linear    │           │Linear │  │Linear │
│  [D x H]    │    →      │[D xH/2]│  │[D xH/2]│
└─────────────┘           └───────┘  └───────┘
    GPU 0                     GPU 0      GPU 1
```

- **传统方式**: 一个 Linear 层的权重矩阵 `[D_in, D_out]` 存在一个 GPU 上
- **Tensor Parallel**: 将权重矩阵切分到多个 GPU，每个 GPU 只存储 `[D_in, D_out/N]`

---

## 2. 为什么需要 Tensor Parallel？

### 2.1 内存瓶颈

现代大模型（如 Diffusion Transformer）的参数量巨大：
- 一个 FFN 层: `dim=4096, hidden_dim=16384` → 约 268M 参数
- 单 GPU 显存不够存储完整模型

### 2.2 计算瓶颈

即使能放入内存，单 GPU 计算速度也有上限。TP 可以实现**近线性加速**：
- N 个 GPU → 理论上 N 倍加速
- 通信开销相对较小（只需要 all-reduce）

---

## 3. 两种核心切分方式

### 3.1 Column Parallel (列切分)

```
权重矩阵 W: [D_in, D_out]

切分方式: 按列切分 (输出维度)

GPU 0: W_0 = W[:, 0:D_out/N]      → 输出前半部分
GPU 1: W_1 = W[:, D_out/N:]       → 输出后半部分

前向计算:
Y = X @ W
→ Y_0 = X @ W_0  (GPU 0)
→ Y_1 = X @ W_1  (GPU 1)
→ Y = [Y_0, Y_1] (concat)
```

**特点**:
- 输入 X 需要**复制**到所有 GPU
- 输出 Y 是**切分**的 (每个 GPU 有部分输出)
- **不需要通信** (计算后直接得到切分结果)

### 3.2 Row Parallel (行切分)

```
权重矩阵 W: [D_in, D_out]

切分方式: 按行切分 (输入维度)

GPU 0: W_0 = W[0:D_in/N, :]       → 处理输入前半部分
GPU 1: W_1 = W[D_in/N:, :]        → 处理输入后半部分

前向计算:
Y = X @ W
→ Y_0 = X_0 @ W_0  (GPU 0)
→ Y_1 = X_1 @ W_1  (GPU 1)
→ Y = Y_0 + Y_1    (all-reduce 求和)
```

**特点**:
- 输入 X 需要**切分**到各个 GPU
- 输出 Y 需要**all-reduce** (通信)
- 每个 GPU 得到完整输出的部分和

---

## 4. 标准配对模式: Column → Row

这是最经典的 TP 模式，用于 MLP 和 Attention：

```
          输入 X (复制)
              │
              ▼
    ┌─────────────────┐
    │ ColumnParallel  │  权重按列切分
    │    Linear       │
    └─────────────────┘
              │
              ▼
       中间结果 (切分)  ← 无需通信
              │
              ▼
    ┌─────────────────┐
    │ RowParallel     │  权重按行切分
    │    Linear       │
    └─────────────────┘
              │
              ▼
       All-Reduce 求和  ← 需要通信
              │
              ▼
          输出 Y (复制)
```

### 优势

1. **最小化通信**: 只有在 Row Parallel 层输出时才需要一次 all-reduce
2. **中间计算并行**: 激活函数等操作可以在切分的数据上独立进行
3. **内存效率**: 每层权重减半，激活值也减半

---

## 5. 数学推导

### 5.1 MLP 的 TP 并行化

标准 MLP:
```
Y = W2 @ act(W1 @ X)
```
其中:
- X: [B, S, D]
- W1: [D, H]  (Column Parallel)
- W2: [H, D]  (Row Parallel)

**并行化推导**:

```
Step 1: W1 按列切分
W1 = [W1_0 | W1_1]  (每个 GPU 有 [D, H/N])

Step 2: 计算 W1 @ X (并行，无通信)
Z_0 = W1_0 @ X  → [B, S, H/N]  (GPU 0)
Z_1 = W1_1 @ X  → [B, S, H/N]  (GPU 1)

Step 3: 激活函数 (并行，无通信)
A_0 = act(Z_0)  (GPU 0)
A_1 = act(Z_1)  (GPU 1)

Step 4: W2 按行切分
W2 = [W2_0]    (每个 GPU 有 [H/N, D])
     [W2_1]

Step 5: 计算 W2 @ A (并行)
Y_0 = W2_0 @ A_0  → [B, S, D]  (GPU 0 的部分和)
Y_1 = W2_1 @ A_1  → [B, S, D]  (GPU 1 的部分和)

Step 6: All-Reduce 求和
Y = Y_0 + Y_1  (需要在 GPU 间通信)
```

### 5.2 Attention 的 TP 并行化

标准 Attention:
```
Q = X @ W_q
K = X @ W_k
V = X @ W_v
Out = Attention(Q, K, V) @ W_o
```

**并行化**:

```
Step 1: QKV 投影 (Column Parallel)
QKV = X @ [W_q | W_k | W_v]

按 head 切分:
- GPU 0: heads 0 ~ num_heads/N
- GPU 1: heads num_heads/N ~ 2*num_heads/N

Step 2: Attention 计算 (并行，无通信)
每个 GPU 独立计算自己负责的 heads

Step 3: Output 投影 (Row Parallel)
Out_partial = Attention_output @ W_o_partial
Out = All-Reduce(Out_partial)
```

---

## 6. 关键约束

为了让 TP 正确工作，某些维度必须能被 `tensor_parallel_size` 整除：

| 维度 | 原因 | 错误示例 |
|------|------|----------|
| `num_heads` | 头数需要平均分配 | `num_heads=30, tp=4` → 30/4=7.5 不行 |
| `num_kv_heads` | KV 头数需要平均分配 | 同上 |
| `hidden_dim` (FFN) | 中间维度需要平均分配 | `hidden_dim=100, tp=4` → 25 OK |

**验证代码**:
```python
assert num_heads % tensor_parallel_size == 0, \
    f"num_heads ({num_heads}) must be divisible by TP size ({tensor_parallel_size})"
```

---

## 7. 通信开销分析

### 通信量计算

每次 all-reduce 的通信量:
```
通信量 = 2 * (N-1)/N * D * sizeof(dtype)

其中:
- N: GPU 数量
- D: 输出维度
- 2: send + receive
```

### 通信频率

在标准 Column→Row 模式下:
- 每个 MLP 块: **1 次** all-reduce
- 每个 Attention 块: **1 次** all-reduce

### 通信 vs 计算

对于大模型，计算量远大于通信量：
```
计算量: O(D * H * S)  (矩阵乘法)
通信量: O(D * S)      (all-reduce)

比值: H (hidden_dim) 通常很大 (数千到数万)
→ 通信开销比例小，效率高
```
