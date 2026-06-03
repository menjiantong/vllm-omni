# Ovis Image Pipeline dtype 修复

## 问题描述

错误信息：
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

发生在 `self.x_embedder(hidden_states)` 时，因为：
- `hidden_states` (来自 text_encoder 输出) 是 `float32`
- `x_embedder` 的权重是 `bfloat16`

## 根本原因

当 `Qwen3Model.from_pretrained` 不传 `torch_dtype` 参数时，模型使用默认的 `torch.float32`，导致 text_encoder 输出 float32 的 embeddings。而 transformer 使用 `od_config.dtype`（如 bfloat16），两者不匹配。

## Ovis 源码的解决方案

Ovis 源码（`temp/Ovis/ovis/model/modeling_ovis.py`）使用 **forward 时转换 dtype** 的方式：

### 关键代码（第 309-311 行）

```python
def merge_multimodal(self, input_ids, pixel_values, grid_thws):
    # text embeddings 来自 LLM 的 word embedding
    multimodal_embeds = self.get_wte()(torch.masked_fill(input_ids, placeholder_token_mask, 0))
    
    if pixel_values is not None:
        # 关键：转换到 multimodal_embeds 的 dtype
        visual_indicator_embeds = self.vte(...).to(dtype=multimodal_embeds.dtype, device=multimodal_embeds.device)
        visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)
        visual_embeds = self.vte(visual_tokens).to(dtype=multimodal_embeds.dtype, device=multimodal_embeds.device)
```

### Ovis 的设计思路

```
visual_tokenizer (ViT)     →  float32 或其他精度
       ↓
vte (VisualEmbedding)      →  可能不同精度
       ↓
.to(dtype=multimodal_embeds.dtype)  ← 关键转换点
       ↓
multimodal_embeds (LLM输入) →  LLM 的精度 (如 bfloat16)
```

**Ovis 不强制所有组件使用相同 dtype，而是在合并多模态 embeddings 时转换到目标 dtype。**

## vllm-omni 的修改

参照 Ovis 源码的设计，在 `_get_ovis_prompt_embeds` 方法中添加 dtype 转换：

### 修改位置 1：`_get_ovis_prompt_embeds` (第 263-265 行)

```python
# Ensure prompt_embeds matches the target dtype (transformer's dtype)
# This handles the case where text_encoder may be in a different dtype
# (e.g., float32 for higher precision) than the transformer (e.g., bfloat16).
prompt_embeds = prompt_embeds.to(dtype=target_dtype)
```

### 修改位置 2：`encode_prompt` (第 299-304 行)

```python
# Use transformer's dtype from config for text_ids to ensure consistency
target_dtype = self.od_config.dtype
text_ids = torch.zeros(prompt_embeds.shape[1], 3)
text_ids[..., 1] = text_ids[..., 1] + torch.arange(prompt_embeds.shape[1])[None, :]
text_ids[..., 2] = text_ids[..., 2] + torch.arange(prompt_embeds.shape[1])[None, :]
text_ids = text_ids.to(device=device, dtype=target_dtype)
```

## 数据流对比

### Ovis 源码

```
ViT (float32) → visual_embeds → .to(dtype=llm.dtype) → LLM (bfloat16)
```

### vllm-omni 修改后

```
text_encoder (float32) → prompt_embeds → .to(dtype=transformer.dtype) → transformer (bfloat16)
```

## 两种方案对比

| 方案 | Ovis 使用 | vllm-omni 使用 | 说明 |
|-----|----------|---------------|------|
| **加载时指定 dtype** | ❌ | ✅ (可选) | text_encoder 内部计算也用目标 dtype |
| **forward 时转换** | ✅ | ✅ | 在合并 embeddings 时转换 dtype |

vllm-omni 现在支持两种方式：
1. 可以在加载时指定 `torch_dtype=od_config.dtype`
2. 即使不指定，forward 时也会转换到正确的 dtype

## 测试建议

两种方式都应该能正常工作：

```python
# 方式1：加载时指定 dtype（可选）
self.text_encoder = Qwen3Model.from_pretrained(
    model, subfolder="text_encoder", torch_dtype=od_config.dtype
)

# 方式2：不指定 dtype，forward 时自动转换
self.text_encoder = Qwen3Model.from_pretrained(
    model, subfolder="text_encoder"
)
```

## 相关文件

### Ovis 源码
- `temp/Ovis/ovis/model/modeling_ovis.py`
  - `merge_multimodal()`: 第 286-317 行，dtype 转换发生的位置

### vllm-omni 修改
- `vllm_omni/diffusion/models/ovis_image/pipeline_ovis_image.py`
  - `_get_ovis_prompt_embeds()`: 第 221-267 行
  - `encode_prompt()`: 第 269-305 行
