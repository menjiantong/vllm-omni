# x_embedder 错误调试指南

## 日志前缀

所有调试日志使用前缀：`----my_debug---`

## 数据流和类型来源追踪

### 1. od_config.dtype 来源
```
位置: vllm_omni/diffusion/data.py:412
默认值: torch.bfloat16
可以通过字符串转换: "bfloat16", "float16", "float32" 等
```

### 2. text_encoder.dtype 来源
```
位置: vllm_omni/diffusion/models/ovis_image/pipeline_ovis_image.py:170
来源: Qwen3Model.from_pretrained() 加载的模型默认 dtype
注意: HuggingFace 模型通常默认是 float32
```

### 3. prompt_embeds.dtype 来源
```
位置: pipeline_ovis_image.py _get_ovis_prompt_embeds 方法
来源: text_encoder.dtype (第250行: dtype = dtype or self.text_encoder.dtype)
```

### 4. latents.dtype 来源
```
位置: pipeline_ovis_image.py:732
来源: prompt_embeds.dtype
```

### 5. hidden_states.dtype 来源
```
位置: ovis_image_transformer.py forward 方法
来源: latents (hidden_states 就是传入的 latents)
```

### 6. x_embedder.weight.dtype 来源
```
位置: ovis_image_transformer.py:410
来源: set_default_torch_dtype(od_config.dtype) 设置的默认 dtype
位置: diffusers_loader.py:310
```

## 类型不匹配问题

**问题根源**: text_encoder.dtype 与 od_config.dtype 可能不一致

```
text_encoder.dtype (来自 HF 模型，通常是 float32)
    ↓
prompt_embeds.dtype
    ↓
latents.dtype
    ↓
hidden_states.dtype (输入到 x_embedder)

vs

od_config.dtype (默认 bfloat16)
    ↓
set_default_torch_dtype()
    ↓
x_embedder.weight.dtype (模型权重)
```

## 添加的调试日志

### 1. OmniDiffusionConfig 初始化 (data.py)
```
----my_debug--- [OmniDiffusionConfig.__post_init__] START
----my_debug--- [OmniDiffusionConfig] model=..., model_class_name=...
----my_debug--- [OmniDiffusionConfig] dtype before conversion: ...
----my_debug--- [OmniDiffusionConfig] dtype after conversion: ...
----my_debug--- [OmniDiffusionConfig] quantization_config=...
----my_debug--- [OmniDiffusionConfig.__post_init__] END
```

### 2. 模型加载 (diffusers_loader.py)
```
----my_debug--- ===== load_model START =====
----my_debug--- [diffusers_loader] od_config.dtype=...
----my_debug--- [diffusers_loader] od_config.model=...
----my_debug--- [diffusers_loader] od_config.quantization_config=...
----my_debug--- [diffusers_loader] set_default_torch_dtype to ...
----my_debug--- [diffusers_loader] transformer.x_embedder.weight dtype=...
----my_debug--- ===== load_model END =====
```

### 3. Pipeline 初始化 (pipeline_ovis_image.py)
```
----my_debug--- [OvisImagePipeline.__init__] START
----my_debug--- [OvisImagePipeline] text_encoder loaded, dtype=...
----my_debug--- [OvisImagePipeline] transformer.x_embedder.weight dtype=...
----my_debug--- [OvisImagePipeline.__init__] END
```

### 4. Transformer 初始化 (ovis_image_transformer.py)
```
----my_debug--- [OvisImageTransformer2DModel.__init__] START
----my_debug--- [Transformer] od_config.dtype=...
----my_debug--- [Transformer] in_channels=..., inner_dim=...
----my_debug--- [Transformer] x_embedder created, weight.dtype=...
```

### 5. Prompt 编码 (pipeline_ovis_image.py)
```
----my_debug--- [pipeline._get_ovis_prompt_embeds] START
----my_debug--- [pipeline] text_encoder.dtype=...
----my_debug--- [pipeline] target dtype=...
----my_debug--- [pipeline] transformer.x_embedder.weight dtype=...
```

### 6. Latents 准备 (pipeline_ovis_image.py)
```
----my_debug--- [pipeline] prepare_latents START
----my_debug--- [pipeline] prompt_embeds.dtype=...
----my_debug--- [pipeline] transformer.x_embedder.weight dtype=...
----my_debug--- [pipeline] latents dtype will be=...
----my_debug--- [pipeline] latents prepared: dtype=...
```

### 7. x_embedder forward (ovis_image_transformer.py)
```
----my_debug--- ===== x_embedder forward START =====
----my_debug--- [INPUT hidden_states] shape=..., dtype=..., device=...
----my_debug--- [x_embedder.weight] shape=..., dtype=..., device=...
----my_debug--- [DIMENSION CHECK] ... MATCH=...
----my_debug--- [DTYPE CHECK] ... MATCH=...
----my_debug--- [DEVICE CHECK] ... MATCH=...
----my_debug--- ===== x_embedder forward END =====
```

## 运行后查看日志

```bash
# 查看所有调试日志
grep "----my_debug---" your_log_file

# 查看 dtype 检查结果
grep "----my_debug---.*MATCH" your_log_file

# 查看 dtype 相关日志
grep "----my_debug---.*dtype" your_log_file
```

## 解决方案

### 方案 1: 统一 dtype (推荐)
修改 `pipeline_ovis_image.py`，使用 transformer 的 dtype:
```python
# 位置: pipeline_ovis_image.py prepare_latents 调用处 (约732行)
# 修改前: dtype=prompt_embeds.dtype
# 修改后:
dtype=self.transformer.x_embedder.weight.dtype
```

### 方案 2: 在 forward 中转换 dtype
修改 `ovis_image_transformer.py`:
```python
# 在 x_embedder 调用前添加
hidden_states = hidden_states.to(self.x_embedder.weight.dtype)
hidden_states = self.x_embedder(hidden_states)
```

### 方案 3: 加载 text_encoder 时指定 dtype
修改 `pipeline_ovis_image.py` 的 `__init__`:
```python
# 修改 text_encoder 加载
self.text_encoder = Qwen3Model.from_pretrained(
    model, subfolder="text_encoder", local_files_only=local_files_only,
    torch_dtype=od_config.dtype  # 添加这行
)
```

## 关键代码位置

| 文件 | 行号 | 说明 |
|------|------|------|
| data.py | 412 | dtype 默认值定义 |
| data.py | 680-697 | dtype 字符串转换 |
| diffusers_loader.py | 310 | set_default_torch_dtype |
| pipeline_ovis_image.py | 170 | text_encoder 加载 |
| pipeline_ovis_image.py | 250 | prompt_embeds dtype 来源 |
| pipeline_ovis_image.py | 732 | latents dtype 设置 |
| ovis_image_transformer.py | 410 | x_embedder 创建 |
| ovis_image_transformer.py | 474 | x_embedder 调用 |
