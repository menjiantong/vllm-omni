# text_encoder dtype 来源和精度问题分析

## 1. text_encoder 的 dtype 来源

### HuggingFace from_pretrained 的 dtype 决定流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    from_pretrained() 调用                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 是否指定了 torch_dtype 参数？                                      │
│   - 是 → 使用指定的 dtype，自动转换权重                             │
│   - 否 → 继续检查                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ config.json 中是否有 torch_dtype 字段？                            │
│   - 是 → 使用 config.json 中的 dtype                               │
│   - 否 → 使用权重文件本身的 dtype                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 权重文件的 dtype

- `.safetensors` 文件：保存时就带有 dtype 信息
- `.bin` 文件：PyTorch pickle 格式，保存时带有 dtype

**大多数 HuggingFace 模型保存为 `float32`**，因为：
1. float32 是训练时的标准格式
2. 保存为 float32 可以避免精度问题
3. 用户可以根据需要自行转换

## 2. config.json 中的 dtype

查看 Qwen 模型的 config.json：

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "torch_dtype": "float32",  // 这里指定了推荐的 dtype
  ...
}
```

**但是**，如果 `torch_dtype` 在 config.json 中是 `float32`，而你不指定 `torch_dtype` 参数，就会加载为 `float32`。

## 3. 精度损失问题

### float32 → bfloat16 转换

| 特性 | float32 | bfloat16 |
|------|---------|----------|
| 位数 | 32 bit | 16 bit |
| 指数位 | 8 bit | 8 bit |
| 尾数位 | 23 bit | 7 bit |
| 动态范围 | ±3.4e38 | ±3.4e38 (相同) |
| 精度 | ~7 位有效数字 | ~2-3 位有效数字 |

**结论**：
- **动态范围相同**：不会出现数值溢出问题
- **精度降低**：从 7 位有效数字降到 2-3 位
- **对 LLM 影响小**：模型参数本身就不是精确值，bfloat16 足够

### float32 → float16 转换

| 特性 | float32 | float16 |
|------|---------|----------|
| 位数 | 32 bit | 16 bit |
| 指数位 | 8 bit | 5 bit |
| 尾数位 | 23 bit | 10 bit |
| 动态范围 | ±3.4e38 | ±65504 |
| 精度 | ~7 位有效数字 | ~3-4 位有效数字 |

**风险**：
- **动态范围小**：数值 > 65504 会溢出
- **需要特别注意**：梯度、激活值可能溢出

### 实际影响

```
                    ┌──────────────────┐
                    │  训练用 float32   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  保存为 float32   │  ← HuggingFace 标准做法
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌────────────┐    ┌────────────┐    ┌────────────┐
    │ 推理 float32│    │ 推理 bfloat16│   │ 推理 float16│
    │ 精度：100% │    │ 精度：~99%  │    │ 精度：~95% │
    │ 显存：最大  │    │ 显存：减半   │    │ 显存：减半  │
    │ 速度：最慢  │    │ 速度：更快   │    │ 速度：更快  │
    └────────────┘    └────────────┘    └────────────┘
```

**对于推理**，bfloat16 通常是最佳选择：
- 精度损失可以忽略
- 显存减半
- 计算速度更快（现代 GPU 对 bfloat16 有优化）

## 4. 其他模型/框架的处理方式

### vLLM (LLM 推理)

```python
# vLLM 在加载模型时强制使用指定的 dtype
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,  # 强制指定
)
```

### Diffusers (Diffusion 模型)

```python
# Diffusers pipeline 也支持 torch_dtype
pipeline = DiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 或 torch.bfloat16
)
```

### Ovis-Image 的 text_encoder

**问题**：之前没有指定 `torch_dtype`，导致使用了模型原始的 float32

**修复后**：
```python
self.text_encoder = Qwen3Model.from_pretrained(
    model, subfolder="text_encoder",
    torch_dtype=od_config.dtype,  # 强制使用配置的 dtype
)
```

## 5. 其他 text_encoder 的处理

查看代码发现，qwen_image 等其他模型的 text_encoder 也没有指定 `torch_dtype`：

```python
# qwen_image/pipeline_qwen_image.py:291
self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model, subfolder="text_encoder", local_files_only=local_files_only
    # 缺少 torch_dtype 参数！
)
```

**这可能是 vllm-omni 的一个普遍问题**，其他模型可能也会有同样的 dtype 不匹配问题。

## 6. 推荐做法

### 最佳实践

```python
# 加载任何 HuggingFace 模型时，都应该指定 torch_dtype
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 或 config.dtype
)
```

### 用户可配置

```yaml
# 配置文件
engine_args:
  dtype: bfloat16  # 用户可以指定 float16, float32, bfloat16
```

### 默认值

- 训练：float32
- 推理：bfloat16（推荐）或 float16
- 低显存：float16 或量化 (int8, int4)
