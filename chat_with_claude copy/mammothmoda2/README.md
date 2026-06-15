# MammothModa2 模型架构文档

## 文档索引

| 文档 | 内容 |
|------|------|
| [01_model_overview.md](01_model_overview.md) | 模型概述、整体架构、支持任务、核心创新点 |
| [02_ar_stage.md](02_ar_stage.md) | AR 阶段详解：类层次、MoE 架构、双词汇表、张量形状 |
| [03_dit_stage.md](03_dit_stage.md) | DiT 阶段详解：扩散流程、Transformer2DModel、Scheduler |
| [04_data_flow.md](04_data_flow.md) | 数据流向：T2I 流程、Understanding 流程、ar2dit 处理器 |
| [05_config.md](05_config.md) | 配置参数：配置类层次、参数详解、Stage 配置文件 |
| [06_rope_tokenizer.md](06_rope_tokenizer.md) | RoPE 与 Tokenizer：3D 位置编码、MammothUTokenizer |

## 快速导航

### 核心组件

- **AR 阶段入口**: `MammothModa2ForConditionalGeneration` (`mammoth_moda2.py:694`)
- **MoE 语言模型**: `MammothModa2Qwen2ForCausalLM` (`mammoth_moda2.py:255`)
- **MoE 路由**: `moe_forward()` (`mammoth_moda2.py:75`)
- **DiT Pipeline**: `MammothModa2DiTPipeline` (`pipeline_mammothmoda2_dit.py:22`)
- **DiT Transformer**: `Transformer2DModel` (`mammothmoda2_dit_model.py:493`)
- **Stage 转换**: `ar2dit()` (`stage_input_processors/mammoth_moda2.py:11`)

### 关键配置

- **顶层配置**: `Mammothmoda2Config` (`configs/mammoth_moda2.py:210`)
- **LLM 配置**: `Mammothmoda2Qwen2_5_VLConfig` (`configs/mammoth_moda2.py:142`)
- **Text 配置**: `Mammothmoda2Qwen2_5_VLTextConfig` (`configs/mammoth_moda2.py:61`)
- **Vision 配置**: `Mammothmoda2Qwen2_5_VLVisionConfig` (`configs/mammoth_moda2.py:20`)

### 张量形状速查

| 组件 | 输入形状 | 输出形状 |
|------|----------|----------|
| embed_tokens | `[batch, seq_len]` | `[batch, seq_len, 8192]` |
| DecoderLayer | `[batch, seq_len, 8192]` | `[batch, seq_len, 8192]` |
| lm_head | `[batch, seq_len, 8192]` | `[batch, seq_len, 184864]` |
| DiT forward | `[1, 16, H/8, W/8]` | `[1, 16, H/8, W/8]` |
| VAE decode | `[1, 16, H/8, W/8]` | `[1, 3, H, W]` |

### 关键参数

```python
# 词汇表
base_vocab_size = 152064
gen_vocab_size = 32800
total_vocab_size = 184864
gen_vocab_start_index = 152064

# 隐藏层
llm_hidden_size = 8192
dit_hidden_size = 2304
vision_hidden_size = 3584

# 层数
num_llm_layers = 80
num_dit_layers = 26
num_vit_layers = 32
```

## 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MammothModa2 完整架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    输入                                                                      │
│     │                                                                       │
│     ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                       AR Stage                                        │ │
│  │  ┌─────────────┐     ┌─────────────────────────────────────────────┐ │ │
│  │  │ Vision Tower│────▶│ MoE Language Model (Qwen2.5-VL)             │ │ │
│  │  │   (ViT)     │     │  ├─ embed_tokens (base vocab)               │ │ │
│  │  │  32 layers  │     │  ├─ gen_embed_tokens (gen vocab)            │ │ │
│  │  │  3584 dim   │     │  ├─ 80 × Mammoth2DecoderLayer               │ │ │
│  │  └─────────────┘     │  │   ├─ self_attn                           │ │ │
│  │                      │  │   ├─ mlp (understanding expert)          │ │ │
│  │                      │  │   └─ gen_mlp (generation expert)         │ │ │
│  │                      │  ├─ lm_head (base logits)                   │ │ │
│  │                      │  └─ gen_head (gen logits)                   │ │ │
│  │                      └─────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│     │                                                                       │
│     │ hidden_states + generated_tokens                                      │
│     ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    ar2dit Processor                                   │ │
│  │  提取 text_condition 和 image_condition embeddings                    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│     │                                                                       │
│     ▼                                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                       DiT Stage                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │ Transformer2DModel (26 layers)                                  │ │ │
│  │  │  ├─ noise_refiner (2 layers): 处理噪声 latent                   │ │ │
│  │  │  ├─ context_refiner (2 layers): 处理文本条件                    │ │ │
│  │  │  └─ layers (26 layers): 主干 Transformer                       │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  │     │                                                                 │ │
│  │     ▼                                                                 │ │
│  │  ┌──────────────────┐                                                │ │
│  │  │ VAE Decoder      │                                                │ │
│  │  │ latent → image   │                                                │ │
│  │  └──────────────────┘                                                │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│     │                                                                       │
│     ▼                                                                       │
│    输出图像                                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 关键代码路径

### T2I 推理入口

```python
# examples/offline_inference/mammothmodal2_preview/run_mammothmoda2_t2i.py
from vllm_omni import LLM, SamplingParams

llm = LLM(
    model="path/to/mammothmoda2",
    trust_remote_code=True,
    stage_config_path="mammoth_moda2.yaml",
)

outputs = llm.generate(prompts, sampling_params)
```

### Understanding 推理入口

```python
# examples/offline_inference/mammothmodal2_preview/run_mammothmoda2_image_summarize.py
llm = LLM(
    model="path/to/mammothmoda2",
    trust_remote_code=True,
    stage_config_path="mammoth_moda2_ar.yaml",  # 单阶段 AR
)
```

## 参考资源

- [vLLM 文档](https://vllm.readthedocs.io/)
- [Qwen2.5-VL 论文](https://arxiv.org/abs/xxxx)
- [Diffusers 文档](https://huggingface.co/docs/diffusers/)
