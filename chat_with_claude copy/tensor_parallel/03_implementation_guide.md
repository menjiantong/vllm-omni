# 为模型添加 Tensor Parallel 支持实现指南

## 概述

本指南详细说明如何为 vLLM Omni 中的 Diffusion 模型添加 Tensor Parallel 支持。

---

## Step 1: 识别需要切分的层

### 1.1 找到所有 Linear 层

首先，检查你的模型中有哪些 `nn.Linear` 层：

```python
# 打印所有 Linear 层
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name}: {module.in_features} -> {module.out_features}")
```

### 1.2 分类 Linear 层

根据层的功能确定切分类型：

| 层类型 | 应使用的并行层 | 原因 |
|--------|---------------|------|
| QKV 投影 | `QKVParallelLinear` | 自动处理 head 切分 |
| Attention 输出 | `RowParallelLinear` | 配对 QKV 的 Column Parallel |
| FFN 第一个投影 (up) | `ColumnParallelLinear` | 扩展维度 |
| FFN 第二个投影 (down) | `RowParallelLinear` | 收缩维度，配对 up |
| LayerNorm 后的投影 | `ColumnParallelLinear` | 通常 |
| 最终输出层 | `ReplicatedLinear` 或不切分 | 根据需求 |
| 嵌入层 | 通常复制或不切分 | 词汇表切分复杂 |

---

## Step 2: 导入并行层

```python
# 从 vLLM 导入并行层
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
)

# 从 vLLM Omni 导入 Attention
from vllm_omni.diffusion.attention.layer import Attention
```

---

## Step 3: 替换 MLP 层

### 3.1 标准 MLP (GELU 激活)

**原始代码**:
```python
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))
```

**TP 版本**:
```python
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Column Parallel: 权重切分为 [dim, hidden_dim/tp_size]
        self.w1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            return_bias=False,  # 通常设为 False
        )
        self.act = nn.GELU()
        
        # Row Parallel: 权重切分为 [hidden_dim/tp_size, dim]
        # input_is_parallel=True 表示输入来自 ColumnParallel，已经是切分的
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,  # 重要！
            return_bias=False,
        )

    def forward(self, x):
        # x: [B, S, dim] (所有 GPU 相同)
        # w1 输出: [B, S, hidden_dim/tp_size] (切分的)
        x = self.w1(x)
        # 激活函数在切分数据上独立计算
        x = self.act(x)
        # w2 输出: [B, S, dim] (all-reduce 后所有 GPU 相同)
        x = self.w2(x)
        return x
```

### 3.2 GLU 变体 (SwiGLU/GEGLU)

**原始代码**:
```python
class GEGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # gate 和 value 两个分支
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # value
        self.act = nn.GELU()
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # output

    def forward(self, x):
        gate = self.act(self.w1(x))
        value = self.w2(x)
        return self.w3(gate * value)
```

**TP 版本**:
```python
class GEGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # 两个 Column Parallel 层
        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, return_bias=False)
        self.w2 = ColumnParallelLinear(dim, hidden_dim, bias=False, return_bias=False)
        self.act = nn.GELU()
        
        # 一个 Row Parallel 层
        self.w3 = RowParallelLinear(
            hidden_dim, dim, 
            bias=False, 
            input_is_parallel=True,  # 输入来自 w1*w2，已是切分的
            return_bias=False
        )

    def forward(self, x):
        # w1, w2 输出都是切分的
        gate = self.act(self.w1(x))  # [B, S, hidden_dim/tp_size]
        value = self.w2(x)           # [B, S, hidden_dim/tp_size]
        # 逐元素乘法在切分数据上进行
        x = gate * value
        # w3 进行 all-reduce
        return self.w3(x)
```

---

## Step 4: 替换 Attention 层

### 4.1 标准 Multi-Head Attention

**原始代码**:
```python
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.to_qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # 简化的 attention
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return self.to_out(out.transpose(1, 2).reshape(B, S, D))
```

**TP 版本**:
```python
from vllm_omni.diffusion.attention.layer import Attention

class TPAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int = None):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads  # MHA
        
        self.head_dim = dim // num_heads
        
        # QKV Parallel: 自动处理 head 切分
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            return_bias=False,
        )
        
        # 使用 vLLM Omni 的 Attention 层
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,      # 本地 head 数
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,                          # Diffusion 通常非因果
            num_kv_heads=self.to_qkv.num_kv_heads, # 本地 KV head 数
        )
        
        # Output projection
        self.to_out = RowParallelLinear(
            dim, dim, 
            bias=False, 
            input_is_parallel=True,
            return_bias=False
        )

    def forward(self, x):
        B, S, D = x.shape
        
        # QKV 投影 (Column Parallel)
        qkv = self.to_qkv(x)
        
        # 分离 Q, K, V
        # 注意: 使用本地 head 数计算大小
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        k_size = kv_size
        v_size = kv_size
        
        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
        
        # 重塑为 attention 格式
        q = q.view(B, S, self.to_qkv.num_heads, self.head_dim)
        k = k.view(B, S, self.to_qkv.num_kv_heads, self.head_dim)
        v = v.view(B, S, self.to_qkv.num_kv_heads, self.head_dim)
        
        # Attention 计算 (每个 GPU 独立处理自己的 heads)
        out = self.attn(q, k, v)
        
        # 重塑并投影 (Row Parallel + all-reduce)
        out = out.view(B, S, -1)
        out = self.to_out(out)
        
        return out
```

### 4.2 关键点说明

```python
# ⚠️ 常见错误: 使用总 head 数而不是本地 head 数

# ❌ 错误
q_size = self.total_num_heads * self.head_dim  # 总 head 数

# ✅ 正确
q_size = self.to_qkv.num_heads * self.head_dim  # 本地 head 数
```

---

## Step 5: 添加 TP 约束验证

### 5.1 添加验证函数

```python
def validate_tensor_parallel_constraints(
    num_heads: int,
    num_kv_heads: int,
    hidden_dim: int,
    tp_size: int,
) -> None:
    """
    验证 TP 约束条件
    
    Args:
        num_heads: 注意力头数
        num_kv_heads: KV 头数 (GQA)
        hidden_dim: FFN 隐藏维度
        tp_size: Tensor Parallel 大小
    """
    if num_heads % tp_size != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by "
            f"tensor_parallel_size ({tp_size}). "
            f"Consider using num_heads that is a multiple of {tp_size}."
        )
    
    if num_kv_heads % tp_size != 0:
        raise ValueError(
            f"num_kv_heads ({num_kv_heads}) must be divisible by "
            f"tensor_parallel_size ({tp_size})."
        )
    
    if hidden_dim is not None and hidden_dim % tp_size != 0:
        raise ValueError(
            f"hidden_dim ({hidden_dim}) must be divisible by "
            f"tensor_parallel_size ({tp_size}) for FFN sharding."
        )
```

### 5.2 在模型初始化时验证

```python
class MyTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ff_hidden_dim: int,
        ...
    ):
        super().__init__()
        
        # 获取 TP 大小
        from vllm.distributed import get_tensor_parallel_world_size
        tp_size = get_tensor_parallel_world_size()
        
        # 验证约束
        validate_tensor_parallel_constraints(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_dim=ff_hidden_dim,
            tp_size=tp_size,
        )
        
        # 继续初始化...
```

---

## Step 6: 处理特殊情况

### 6.1 层归一化 (LayerNorm/RMSNorm)

```python
# LayerNorm 通常不切分，在复制的数据上操作
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 复制到所有 GPU

    def forward(self, x):
        # x: [B, S, dim] (复制的)
        # 在复制的数据上计算归一化
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

### 6.2 位置编码

```python
# 位置编码通常复制
class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        # 预计算的频率，复制到所有 GPU
        self.register_buffer(
            "freqs", 
            self._compute_freqs(dim, max_seq_len)
        )
    
    def forward(self, x, seq_len):
        # 所有 GPU 使用相同的频率
        return apply_rotary_emb(x, self.freqs[:seq_len])
```

### 6.3 跨 GPU 不操作的层

```python
# 某些层需要全局信息，需要特殊处理
class AdaptiveLayerNorm(nn.Module):
    """自适应 LayerNorm，用于条件生成"""
    
    def __init__(self, dim: int, condition_dim: int):
        super().__init__()
        # 这些投影需要特殊处理
        # 通常保持复制或使用 ReplicatedLinear
        self.norm = nn.LayerNorm(dim)
        self.linear = ReplicatedLinear(condition_dim, dim, bias=True)
    
    def forward(self, x, condition):
        # 在复制的数据上操作
        scale_shift = self.linear(condition)
        return self.norm(x) * (1 + scale_shift[:, 0]) + scale_shift[:, 1]
```

---

## Step 7: 完整示例

### 7.1 完整的 Transformer Block

```python
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
)
from vllm_omni.diffusion.attention.layer import Attention

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ff_hidden_dim: int,
    ):
        super().__init__()
        
        # Layer Norm (复制)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Attention
        self.head_dim = dim // num_heads
        self.attn_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            return_bias=False,
        )
        self.attn = Attention(
            num_heads=self.attn_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim ** 0.5),
            causal=False,
            num_kv_heads=self.attn_qkv.num_kv_heads,
        )
        self.attn_out = RowParallelLinear(
            dim, dim, bias=False, input_is_parallel=True, return_bias=False
        )
        
        # FFN
        self.ff_up = ColumnParallelLinear(
            dim, ff_hidden_dim, bias=False, return_bias=False
        )
        self.ff_act = nn.GELU()
        self.ff_down = RowParallelLinear(
            ff_hidden_dim, dim, bias=False, 
            input_is_parallel=True, return_bias=False
        )

    def forward(self, x):
        # x: [B, S, dim] (复制)
        
        # Attention 分支
        residual = x
        x = self.norm1(x)  # 在复制数据上
        
        # QKV (Column Parallel)
        qkv = self.attn_qkv(x)
        q_size = self.attn_qkv.num_heads * self.head_dim
        kv_size = self.attn_qkv.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        
        # Reshape for attention
        B, S, _ = q.shape
        q = q.view(B, S, self.attn_qkv.num_heads, self.head_dim)
        k = k.view(B, S, self.attn_qkv.num_kv_heads, self.head_dim)
        v = v.view(B, S, self.attn_qkv.num_kv_heads, self.head_dim)
        
        # Attention (并行)
        attn_out = self.attn(q, k, v)
        attn_out = attn_out.view(B, S, -1)
        
        # Output projection (Row Parallel + all-reduce)
        x = self.attn_out(attn_out)
        x = residual + x  # 残差连接
        
        # FFN 分支
        residual = x
        x = self.norm2(x)
        
        # Up projection (Column Parallel)
        x = self.ff_up(x)
        x = self.ff_act(x)
        
        # Down projection (Row Parallel + all-reduce)
        x = self.ff_down(x)
        x = residual + x
        
        return x  # [B, S, dim] (复制)
```

---

## Step 8: 测试验证

### 8.1 单元测试

```python
import torch
import torch.distributed as dist
from vllm.distributed import initialize_tensor_parallel

def test_tp_correctness():
    """测试 TP 结果与单 GPU 一致"""
    
    # 初始化分布式
    dist.init_process_group(backend="nccl")
    tp_size = 2
    initialize_tensor_parallel(tp_size)
    
    # 创建模型
    model = MyTransformer(dim=256, num_heads=8, ff_hidden_dim=1024)
    
    # 创建输入
    x = torch.randn(1, 16, 256).cuda()
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 验证所有 GPU 输出相同
    outputs = [torch.zeros_like(output) for _ in range(tp_size)]
    dist.all_gather(outputs, output)
    
    for i in range(1, tp_size):
        assert torch.allclose(outputs[0], outputs[i], atol=1e-5), \
            f"GPU 0 and GPU {i} outputs differ!"
    
    print("TP correctness test passed!")
```

### 8.2 E2E 测试

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

def test_e2e_tp():
    """端到端测试"""
    
    # 单 GPU 基准
    omni_single = Omni(model="your-model")
    output_single = omni_single.generate(
        "a test prompt",
        OmniDiffusionSamplingParams(num_inference_steps=20),
    )
    
    # TP=2
    parallel_config = DiffusionParallelConfig(tensor_parallel_size=2)
    omni_tp = Omni(model="your-model", parallel_config=parallel_config)
    output_tp = omni_tp.generate(
        "a test prompt",
        OmniDiffusionSamplingParams(num_inference_steps=20),
    )
    
    # 验证输出质量相似
    # 注意: 由于浮点误差，不会完全相同
    # 但视觉质量应该一致
    
    # 验证内存使用减少
    print(f"Single GPU memory: {get_memory_usage(omni_single)}")
    print(f"TP=2 memory per GPU: {get_memory_usage(omni_tp)}")
    
    # 验证速度提升
    # (通常需要多次运行取平均)
```

### 8.3 命令行测试

```bash
# 测试 TP=2
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Your-org/your-model \
    --prompt "a cup of coffee on the table" \
    --tensor-parallel-size 2 \
    --output "tp2_output.png"

# 检查日志中的内存和时间信息
```

---

## 检查清单

添加 TP 支持后，确认以下事项：

- [ ] 所有 MLP 层使用 `ColumnParallelLinear` + `RowParallelLinear` 配对
- [ ] Attention QKV 使用 `QKVParallelLinear`
- [ ] Attention 输出使用 `RowParallelLinear` 且 `input_is_parallel=True`
- [ ] 使用 `self.to_qkv.num_heads` (本地) 而不是 `total_num_heads`
- [ ] 添加了 TP 约束验证
- [ ] LayerNorm 和位置编码保持复制
- [ ] 测试通过: 输出一致性、内存减少、速度提升
