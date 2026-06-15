# Tensor Parallel 算法逻辑与调用过程

## 1. vLLM Omni 的 TP 架构

### 1.1 核心组件层级

```
┌─────────────────────────────────────────────────────────────┐
│                     用户 API 层                              │
│  Omni.generate() / text_to_image.py                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   并行配置层                                  │
│  DiffusionParallelConfig(tensor_parallel_size=N)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   模型构建层                                  │
│  Model.__init__() → 使用 Parallel Layers                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   并行层实现                                  │
│  ColumnParallelLinear, RowParallelLinear, QKVParallelLinear │
│  (来自 vllm.model_executor.layers.linear)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   通信原语层                                  │
│  tensor_model_parallel_all_reduce()                         │
│  (基于 NCCL 的 all-reduce)                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Parallel Layers 算法详解

### 2.1 ColumnParallelLinear

```python
class ColumnParallelLinear(nn.Module):
    """
    权重按列切分 (输出维度)
    
    权重形状: [in_features, out_features]
    切分后:   [in_features, out_features / tp_size]  (每个 GPU)
    """
    
    def __init__(self, input_size, output_size, ...):
        # 获取 TP 信息
        self.tp_size = get_tensor_parallel_world_size()
        self.tp_rank = get_tensor_parallel_rank()
        
        # 计算本地输出维度
        self.output_size_per_partition = output_size // self.tp_size
        
        # 创建本地权重 (只保存 1/tp_size 的输出维度)
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition, input_size
        ))
    
    def forward(self, x):
        """
        x: [batch, seq, input_size] (所有 GPU 相同)
        output: [batch, seq, output_size/tp_size] (每个 GPU 不同)
        """
        # 直接计算本地部分
        output = F.linear(x, self.weight)
        return output  # 输出是切分的
```

**算法流程**:
```
输入 X: [B, S, D_in]  (复制到所有 GPU)
│
├── GPU 0: Y_0 = X @ W_0^T  → [B, S, D_out/N]
├── GPU 1: Y_1 = X @ W_1^T  → [B, S, D_out/N]
├── GPU 2: Y_2 = X @ W_2^T  → [B, S, D_out/N]
└── GPU 3: Y_3 = X @ W_3^T  → [B, S, D_out/N]

输出: 每个 GPU 有 [B, S, D_out/N] 的切分结果
通信: 无
```

### 2.2 RowParallelLinear

```python
class RowParallelLinear(nn.Module):
    """
    权重按行切分 (输入维度)
    
    权重形状: [in_features, out_features]
    切分后:   [in_features / tp_size, out_features]  (每个 GPU)
    """
    
    def __init__(self, input_size, output_size, input_is_parallel=False, ...):
        self.tp_size = get_tensor_parallel_world_size()
        self.tp_rank = get_tensor_parallel_rank()
        self.input_is_parallel = input_is_parallel
        
        # 计算本地输入维度
        self.input_size_per_partition = input_size // self.tp_size
        
        # 创建本地权重
        self.weight = nn.Parameter(torch.empty(
            output_size, self.input_size_per_partition
        ))
    
    def forward(self, x):
        """
        如果 input_is_parallel=True:
          x: [batch, seq, input_size/tp_size] (切分的)
        否则:
          x: [batch, seq, input_size] (复制的，需要先切分)
        
        output: [batch, seq, output_size] (all-reduce 后相同)
        """
        if not self.input_is_parallel:
            # 需要先切分输入
            x = x[..., self.input_size_per_partition * self.tp_rank:
                     self.input_size_per_partition * (self.tp_rank + 1)]
        
        # 计算本地部分和
        output_parallel = F.linear(x, self.weight)
        
        # All-Reduce 求和
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output  # 输出是完整的
```

**算法流程**:
```
输入 X: [B, S, D_in/N]  (切分的，来自 ColumnParallel)
│
├── GPU 0: Y_0 = X_0 @ W_0  → [B, S, D_out] (部分和)
├── GPU 1: Y_1 = X_1 @ W_1  → [B, S, D_out] (部分和)
├── GPU 2: Y_2 = X_2 @ W_2  → [B, S, D_out] (部分和)
└── GPU 3: Y_3 = X_3 @ W_3  → [B, S, D_out] (部分和)
          │
          ▼
    All-Reduce Sum
          │
          ▼
输出 Y: [B, S, D_out]  (所有 GPU 相同)
通信: 一次 all-reduce
```

### 2.3 QKVParallelLinear

```python
class QKVParallelLinear(nn.Module):
    """
    专门用于 Attention QKV 投影
    自动处理 head 的切分和复制
    """
    
    def __init__(self, hidden_size, head_size, 
                 total_num_heads, total_num_kv_heads, ...):
        self.tp_size = get_tensor_parallel_world_size()
        self.tp_rank = get_tensor_parallel_rank()
        
        # 计算每个 GPU 的 head 数
        self.num_heads = total_num_heads // self.tp_size
        self.num_kv_heads = total_num_kv_heads // self.tp_size
        
        # 本地权重大小
        # Q, K, V 各有自己的 head 数
        q_proj_size = self.num_heads * head_size
        kv_proj_size = self.num_kv_heads * head_size
        total_proj_size = q_proj_size + 2 * kv_proj_size
        
        self.weight = nn.Parameter(torch.empty(
            total_proj_size, hidden_size
        ))
        
        # 保存属性供后续使用
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
    
    def forward(self, x):
        """
        x: [batch, seq, hidden_size]
        output: [batch, seq, (num_q + num_k + num_v) * head_size / tp_size]
        """
        output = F.linear(x, self.weight)
        return output
```

**切分示意图**:
```
Total Heads: 32, TP=4 → 每个 GPU 8 heads

GPU 0: heads 0-7   (Q, K, V 各 8 个 head)
GPU 1: heads 8-15
GPU 2: heads 16-23
GPU 3: heads 24-31

输出切分:
每个 GPU 输出: [B, S, (8 + 8 + 8) * head_dim]
             = [B, S, 24 * head_dim]
```

---

## 3. 完整前向传播调用流程

### 3.1 MLP Block 示例

```python
# 初始化阶段
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Column Parallel: 权重 [dim, hidden_dim/N]
        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        # Row Parallel: 权重 [hidden_dim/N, dim]
        self.w2 = RowParallelLinear(hidden_dim, dim, 
                                     input_is_parallel=True, bias=False)
    
    def forward(self, x):
        # Step 1: Column Parallel (无通信)
        # x: [B, S, dim] (所有 GPU 相同)
        # h: [B, S, hidden_dim/N] (每个 GPU 不同)
        h = self.w1(x)
        
        # Step 2: 激活函数 (并行，无通信)
        h = self.act(h)
        
        # Step 3: Row Parallel (all-reduce 通信)
        # h: [B, S, hidden_dim/N] (切分的)
        # out: [B, S, dim] (all-reduce 后所有 GPU 相同)
        out = self.w2(h)
        
        return out
```

**调用序列图**:
```
时间线 →

GPU 0: [w1(x)] → [act(h)] → [w2(h)] ─┬─→ [all-reduce] → [output]
GPU 1: [w1(x)] → [act(h)] → [w2(h)] ─┤
GPU 2: [w1(x)] → [act(h)] → [w2(h)] ─┤
GPU 3: [w1(x)] → [act(h)] → [w2(h)] ─┘

       ├──── 并行计算 ────┤ ├─通信─┤
```

### 3.2 Attention Block 示例

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()
        self.head_dim = dim // num_heads
        
        # QKV Parallel: 自动处理 head 切分
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
        )
        
        # Attention 层
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,  # 本地 head 数
            head_size=self.head_dim,
            num_kv_heads=self.to_qkv.num_kv_heads,  # 本地 KV head 数
            ...
        )
        
        # Output projection (Row Parallel)
        self.to_out = RowParallelLinear(
            dim, dim, input_is_parallel=True, bias=False
        )
    
    def forward(self, x):
        # x: [B, S, dim] (复制)
        
        # Step 1: QKV 投影 (Column Parallel, 无通信)
        qkv = self.to_qkv(x)
        # qkv: [B, S, (local_q + local_k + local_v) * head_dim]
        
        # Step 2: 分离 Q, K, V
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        # q: [B, S, local_num_heads * head_dim]
        # k: [B, S, local_num_kv_heads * head_dim]
        # v: [B, S, local_num_kv_heads * head_dim]
        
        # Step 3: Attention 计算 (并行，无通信)
        # 每个 GPU 只计算自己负责的 heads
        out = self.attn(q, k, v)
        # out: [B, S, local_num_heads * head_dim]
        
        # Step 4: Output 投影 (Row Parallel, all-reduce)
        out = self.to_out(out)
        # out: [B, S, dim] (all-reduce 后完整)
        
        return out
```

---

## 4. 初始化与权重加载

### 4.1 权重切分逻辑

```python
def load_weights(model, checkpoint_path):
    """加载预训练权重并切分"""
    state_dict = torch.load(checkpoint_path)
    
    for name, param in model.named_parameters():
        if ' ColumnParallelLinear' in str(type(param)):
            # 列切分: 取对应的列
            tp_rank = get_tensor_parallel_rank()
            tp_size = get_tensor_parallel_world_size()
            
            # 计算切分索引
            out_features = state_dict[name].shape[0]
            chunk_size = out_features // tp_size
            start = tp_rank * chunk_size
            end = start + chunk_size
            
            # 加载切分后的权重
            param.data.copy_(state_dict[name][start:end, :])
            
        elif 'RowParallelLinear' in str(type(param)):
            # 行切分: 取对应的行
            in_features = state_dict[name].shape[1]
            chunk_size = in_features // tp_size
            start = tp_rank * chunk_size
            end = start + chunk_size
            
            param.data.copy_(state_dict[name][:, start:end])
            
        else:
            # 非并行层: 直接复制
            param.data.copy_(state_dict[name])
```

### 4.2 Tensor Parallel 初始化

```python
def initialize_tensor_parallel(tp_size):
    """初始化 TP 通信组"""
    # 获取世界大小和 rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 创建 TP 通信组
    # 例如: world_size=8, tp_size=4 → 2 个 TP 组
    # TP 组 0: ranks [0,1,2,3]
    # TP 组 1: ranks [4,5,6,7]
    
    num_tp_groups = world_size // tp_size
    
    for i in range(num_tp_groups):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)
        
        if rank in ranks:
            _TENSOR_PARALLEL_GROUP = group
            _TENSOR_PARALLEL_RANK = rank % tp_size
            _TENSOR_PARALLEL_WORLD_SIZE = tp_size
```

---

## 5. 通信原语

### 5.1 All-Reduce 实现

```python
def tensor_model_parallel_all_reduce(input_):
    """
    在 TP 组内执行 all-reduce sum
    
    使用 NCCL 后端优化
    """
    # 获取 TP 通信组
    group = get_tensor_parallel_group()
    
    # All-reduce sum
    dist.all_reduce(input_, group=group)
    
    return input_
```

### 5.2 通信优化

```python
# vLLM 中的优化: 使用自定义 CUDA kernel 减少 kernel launch 开销

class FusedAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group):
        # 前向: 直接 all-reduce
        dist.all_reduce(input, group=group)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向: 梯度也需要 all-reduce
        dist.all_reduce(grad_output, group=group)
        return grad_output, None
```

---

## 6. 完整调用流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 用户代码                                                      │
│    parallel_config = DiffusionParallelConfig(tp_size=2)         │
│    omni = Omni(model="...", parallel_config=parallel_config)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 初始化 TP                                                     │
│    initialize_tensor_parallel(tp_size=2)                        │
│    - 创建 TP 通信组                                               │
│    - 设置 rank 和 world_size                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 构建模型                                                      │
│    model = Transformer(...)                                     │
│    - 使用 ColumnParallelLinear                                   │
│    - 使用 RowParallelLinear                                      │
│    - 使用 QKVParallelLinear                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 加载权重 (切分)                                                │
│    load_weights(model, checkpoint)                              │
│    - ColumnParallel: 切分列维度                                   │
│    - RowParallel: 切分行维度                                      │
│    - 每个 GPU 加载 1/tp_size 的权重                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 前向传播                                                      │
│    output = model(input)                                        │
│                                                                  │
│    每个 Block:                                                   │
│    ├─ ColumnParallel → 切分输出 (无通信)                          │
│    ├─ 激活函数 → 并行计算 (无通信)                                 │
│    └─ RowParallel → all-reduce (通信)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. 输出                                                          │
│    所有 GPU 得到相同的完整输出                                     │
└─────────────────────────────────────────────────────────────────┘
```
