# OvisImage 模型量化功能设计方案

## 一、背景与目标

### 1.1 概述
为 OvisImage 文本到图像扩散模型添加量化支持，以减少显存占用并提升推理效率。

### 1.2 当前问题
- `ovis_image_transformer.py` 没有量化支持
- 所有 Linear 层缺少 `quant_config` 参数传递
- 使用 `disable_tp=True` 与量化不兼容
- 缺少 `prefix` 参数用于权重加载和量化层识别

### 1.3 参考实现
Flux 模型 (`flux_transformer.py`) 已有完整量化实现，OvisImage 架构与 Flux 相似，可直接参考。

---

## 二、量化策略设计

### 2.1 层级量化决策

| 层类型 | 是否量化 | 原因 |
|--------|----------|------|
| transformer_blocks (双流块) | **否** | Flux Issue #2728：FP8 导致输出噪声 |
| single_transformer_blocks (单流块) | **是** | 占大部分计算量（27层 vs 6层） |
| AdaLayerNorm 调制层 | **否** | 调制参数影响残差流，需保持精度 |
| x_embedder / context_embedder | **否** | 输入嵌入层保持全精度 |
| norm_out / proj_out | **否** | 最终输出层保持全精度 |

### 2.2 预期效果
- 显存节省：约 40-50%（仅单流块量化）
- 生成质量：与 FP16 基本一致（参考 Flux 验证结果）

---

## 三、详细修改方案

### 3.1 修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `ovis_image_transformer.py` | 主要修改 | 添加量化支持，约 150 行修改 |
| `pipeline_ovis_image.py` | 小修改 | 传递 quant_config 参数，1-2 行 |

### 3.2 OvisImageAttention 类修改

**位置**: `ovis_image_transformer.py` 第 40-166 行

**修改内容**:
1. 添加 `quant_config` 和 `prefix` 参数
2. 移除 `disable_tp=True`
3. `to_qkv` / `add_kv_proj` 传递 `quant_config`
4. `to_out[0]` / `to_add_out` 改为 `RowParallelLinear`
5. 添加 FP8 `contiguous()` 处理

```python
# 关键修改示例
self.to_qkv = QKVParallelLinear(
    hidden_size=query_dim,
    head_size=self.head_dim,
    total_num_heads=self.heads,
    bias=bias,
    quant_config=quant_config,  # 新增
    prefix=f"{prefix}.to_qkv",  # 新增
    # 移除 disable_tp=True
)
```

### 3.3 新增 OvisImageFeedForward 类

**原因**: diffusers 的 `FeedForward` 不支持量化和 TP，需要自定义实现。

**实现要点**:
- 使用 `MergedColumnParallelLinear` 处理 SwiGLU 双分支
- 支持 `quant_config` 和 `prefix` 参数
- 与现有权重格式兼容

### 3.4 OvisImageTransformerBlock 类修改

**修改内容**:
1. 添加 `quant_config` 和 `prefix` 参数
2. 所有子模块传递 `quant_config=None`（双流块保持全精度）
3. 使用自定义 `OvisImageFeedForward`

### 3.5 OvisImageSingleTransformerBlock 类修改

**修改内容**:
1. 添加 `quant_config` 和 `prefix` 参数
2. `norm` 调制层传递 `quant_config=None`
3. `proj_mlp` / `proj_out` 使用 `ReplicatedLinear` 支持量化
4. `attn` 传递 `quant_config` 支持量化

### 3.6 OvisImageTransformer2DModel 类修改

**修改内容**:
1. 添加 `quant_config` 参数
2. `transformer_blocks` 传递 `quant_config=None`
3. `single_transformer_blocks` 传递 `quant_config`
4. 更新 `load_weights` 支持新的线性层结构

### 3.7 pipeline_ovis_image.py 修改

```python
# 第 182 行修改
self.transformer = OvisImageTransformer2DModel(
    od_config=od_config,
    quant_config=od_config.quantization_config,  # 新增
)
```

---

## 四、关键代码路径

### 4.1 需要导入的新模块

```python
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm_omni.diffusion.layers.adalayernorm import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
```

### 4.2 FP8 兼容性处理

在 forward 方法中添加 `contiguous()` 调用：

| 位置 | 处理方式 |
|------|----------|
| `to_qkv(hidden_states)` | `hidden_states.contiguous()` |
| `add_kv_proj(encoder_hidden_states)` | `encoder_hidden_states.contiguous()` |
| `to_out[0](hidden_states)` | `hidden_states.contiguous()` |
| `to_add_out(encoder_hidden_states)` | `encoder_hidden_states.contiguous()` |

### 4.3 权重加载兼容

保持现有的 `stacked_params_mapping`，添加 FeedForward 权重映射：

```python
# diffusers 格式 -> 新格式映射
ff_weight_mapping = [
    ("ff.net.0.proj", "ff.linear_in"),
    ("ff.net.2", "ff.linear_out"),
]
```

---

## 五、实现步骤

### Phase 1: 基础组件 (OvisImageFeedForward)
1. 添加 `OvisImageSwiGLU` 类
2. 添加 `OvisImageFeedForward` 类

### Phase 2: 注意力层修改
1. 修改 `OvisImageAttention.__init__`
2. 修改 `OvisImageAttention.forward`

### Phase 3: Block 层修改
1. 修改 `OvisImageTransformerBlock`
2. 修改 `OvisImageSingleTransformerBlock`

### Phase 4: 主模型修改
1. 修改 `OvisImageTransformer2DModel.__init__`
2. 更新 `load_weights` 方法
3. 修改 `pipeline_ovis_image.py`

### Phase 5: 测试验证
1. 权重加载测试
2. 前向传播测试
3. 生成质量对比

---

## 六、验证方案

### 6.1 功能验证
- 权重加载成功，无 missing/unexpected keys
- 无量化时输出与原实现一致
- FP8/INT8 量化时推理正常

### 6.2 质量验证
- 使用相同 seed 和 prompt 对比生成图像
- PSNR/SSIM 指标在可接受范围

### 6.3 性能验证
- 显存占用降低约 40%
- 推理速度无明显下降

---

## 七、参考文件

| 文件 | 用途 |
|------|------|
| `flux_transformer.py` | 主要参考实现 |
| `adalayernorm.py` | 支持 prefix 的 AdaLayerNorm |
| `chat_with_claude/ovis_image/ovis_image_quantization_design.md` | 之前的设计文档 |
