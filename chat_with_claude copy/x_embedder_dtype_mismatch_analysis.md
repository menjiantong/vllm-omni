# x_embedder DTYPE 不匹配问题分析报告

## 问题确认

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

**是的，Float = Float32**，这是 PyTorch 的命名约定：
- `Float` = `torch.float32` (32位浮点)
- `Half` = `torch.float16` (16位浮点)
- `BFloat16` = `torch.bfloat16` (Brain Float 16)

## 已应用的修复

### 修复位置: `pipeline_ovis_image.py:178-181`

```python
# 修改后
self.text_encoder = Qwen3Model.from_pretrained(
    model, subfolder="text_encoder", local_files_only=local_files_only,
    torch_dtype=od_config.dtype  # 添加这行，使用 od_config 的 dtype
)
```

这样 `text_encoder` 会使用 `od_config.dtype` (默认 bfloat16)，与 `transformer.x_embedder.weight` 一致。

## 修复后的数据流

```
od_config.dtype = torch.bfloat16 (默认值)
    │
    ├──► text_encoder 加载时使用 torch_dtype=od_config.dtype
    │         └──► text_encoder.dtype = torch.bfloat16
    │
    ├──► set_default_torch_dtype(od_config.dtype)
    │         └──► transformer.x_embedder.weight.dtype = torch.bfloat16
    │
    └──► prompt_embeds.dtype = text_encoder.dtype = torch.bfloat16
              └──► latents.dtype = prompt_embeds.dtype = torch.bfloat16
                        └──► hidden_states.dtype = torch.bfloat16
                                  └──► x_embedder(hidden_states) ✓ dtype 匹配！
```

## 如何指定不同的 dtype

### 方法 1: 配置文件 (YAML)

```yaml
stages:
  - stage_id: 0
    stage_type: diffusion
    model: AIDC-AI/Ovis-Image-7B
    engine_args:
      dtype: float16  # 或 bfloat16, float32
```

### 方法 2: 命令行参数

```bash
python -m vllm_omni.entrypoints.openai.api_server \
    --model AIDC-AI/Ovis-Image-7B \
    --dtype float16 \
    ...
```

### 方法 3: API 调用时指定

```python
from vllm_omni import AsyncOmniEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig

od_config = OmniDiffusionConfig(
    model="AIDC-AI/Ovis-Image-7B",
    dtype=torch.float16,  # 指定 dtype
)
```

## 原始分析（供参考）

### 类型流向分析（修复前）

| 组件 | dtype | 来源 |
|------|-------|------|
| `od_config.dtype` | `bfloat16` | 默认值 |
| `text_encoder` | `float32` | HuggingFace 模型原始 dtype |
| `transformer.x_embedder.weight` | `bfloat16` | `set_default_torch_dtype` |
| `prompt_embeds` | `float32` | 继承自 `text_encoder.dtype` |
| `latents` | `float32` | 继承自 `prompt_embeds.dtype` |
| `hidden_states` | `float32` | 就是 `latents` |

**问题**：`hidden_states (float32)` 与 `x_embedder.weight (bfloat16)` 不匹配！

### 修复后的类型流向

| 组件 | dtype | 来源 |
|------|-------|------|
| `od_config.dtype` | `bfloat16` | 默认值 |
| `text_encoder` | `bfloat16` | 使用 `torch_dtype=od_config.dtype` |
| `transformer.x_embedder.weight` | `bfloat16` | `set_default_torch_dtype` |
| `prompt_embeds` | `bfloat16` | 继承自 `text_encoder.dtype` |
| `latents` | `bfloat16` | 继承自 `prompt_embeds.dtype` |
| `hidden_states` | `bfloat16` | 就是 `latents` |

**结果**：dtype 一致，问题解决！
