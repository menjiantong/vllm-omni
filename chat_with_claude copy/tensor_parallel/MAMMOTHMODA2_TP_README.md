# MammothModa2 Tensor Parallel 实现说明

## 改造方式

基于现有文件直接改造，不新增独立文件：
- `mammothmoda2_dit_model.py` - 添加 TP 支持
- `pipeline_mammothmoda2_dit.py` - 添加 TP 权重加载

## 核心改动

### 1. LuminaFeedForward (SwiGLU)

**TP 模式 (tp_size > 1)**:
```python
self.w13 = MergedColumnParallelLinear(dim, [inner_dim, inner_dim])  # gate + up
self.act = SiluAndMul()
self.w2 = RowParallelLinear(inner_dim, dim, input_is_parallel=True)  # down
```

**非 TP 模式 (tp_size = 1)**:
```python
self.linear_1 = nn.Linear(dim, inner_dim)  # gate
self.linear_2 = nn.Linear(inner_dim, dim)  # down
self.linear_3 = nn.Linear(dim, inner_dim)  # up
```

### 2. Attention

**TP 模式**: 使用 `TPAttention`
- `QKVParallelLinear` 合并 Q/K/V
- `RowParallelLinear` 输出投影

**非 TP 模式**: 使用 diffusers `Attention`
- `to_q`, `to_k`, `to_v`, `to_out`

### 3. TransformerBlock

根据 `tp_size` 自动选择:
- `TPAttention` (TP 模式)
- `diffusers.Attention` (非 TP 模式)

### 4. TP 约束验证

```python
validate_mammothmoda2_tp_constraints(
    dim=hidden_size,
    num_heads=num_attention_heads,
    num_kv_heads=num_kv_heads,
    ffn_inner_dim=ffn_inner_dim,
    tensor_parallel_size=tp_size,
)
```

### 5. 权重加载

`Transformer2DModel.load_weights`:
```python
stacked_params_mapping = [
    (".feed_forward.w13.", ".feed_forward.linear_1.", 0),  # gate
    (".feed_forward.w13.", ".feed_forward.linear_3.", 1),  # up
    (".attn.to_qkv.", ".attn.to_q.", "q"),  # TP only
    (".attn.to_qkv.", ".attn.to_k.", "k"),  # TP only
    (".attn.to_qkv.", ".attn.to_v.", "v"),  # TP only
]
```

### 6. HSDP 支持

```python
_hsdp_shard_conditions = [_is_transformer_block]
packed_modules_mapping = {
    "feed_forward.w13": ["feed_forward.linear_1", "feed_forward.linear_3"],
    "attn.to_qkv": ["attn.to_q", "attn.to_k", "attn.to_v"],
}
```

## 默认配置 TP 支持

| 参数 | 值 |
|------|-----|
| hidden_size | 2304 |
| num_attention_heads | 24 |
| num_kv_heads | 8 |
| ffn_inner_dim | 9216 |

**支持的 TP sizes**: `[1, 2, 4, 8]`

## 使用方式

```python
from vllm_omni.diffusion.data import DiffusionParallelConfig

parallel_config = DiffusionParallelConfig(tensor_parallel_size=2)
# 模型会自动启用 TP 模式
```

## 权重映射

| 原始权重 | TP 权重 | 切分 |
|---------|--------|------|
| feed_forward.linear_1.weight | feed_forward.w13.weight | 列切分 |
| feed_forward.linear_3.weight | feed_forward.w13.weight | 列切分 |
| feed_forward.linear_2.weight | feed_forward.w2.weight | 行切分 |
| attn.to_q/k/v.weight | attn.to_qkv.weight | 列切分 (TP only) |
| attn.to_out.0.weight | attn.to_out.0.weight | 行切分 (TP only) |
