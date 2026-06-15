# MammothModa2 模型架构

## 1. 模型概述

MammothModa2 是一个多模态生成模型，支持文本理解和图像生成任务。采用 **两阶段架构**：

```
┌─────────────────────────────────────────────────────────────────┐
│                     MammothModa2 整体架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: 文本 + 可选图像                                            │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              AR Stage (自回归阶段)                        │   │
│  │  ┌─────────────┐    ┌─────────────────────────────────┐ │   │
│  │  │ Vision Tower │───▶│ MoE Language Model (Qwen2.5-VL) │ │   │
│  │  │  (ViT)       │    │  - Understanding Expert (mlp)   │ │   │
│  │  └─────────────┘    │  - Generation Expert (gen_mlp)   │ │   │
│  │                      └─────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         │ 输出: 文本 logits + 图像 token hidden states           │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ar2dit Processor                            │   │
│  │  提取 text_condition 和 image_condition embeddings       │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              DiT Stage (扩散阶段)                         │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │ Transformer2DModel                                │   │   │
│  │  │  - noise_refiner: 处理噪声 latent                 │   │   │
│  │  │  - context_refiner: 处理文本条件                  │   │   │
│  │  │  - layers: 主干 Transformer blocks               │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │         │                                                │   │
│  │         ▼                                                │   │
│  │  ┌──────────────────┐                                   │   │
│  │  │ VAE Decoder      │                                   │   │
│  │  │ latent -> image  │                                   │   │
│  │  └──────────────────┘                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  输出: 生成的图像                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 支持的任务模式

| 模式 | Stage | 输入 | 输出 | 配置文件 |
|------|-------|------|------|----------|
| T2I (Text-to-Image) | AR → DiT | 文本 | 图像 | `mammoth_moda2.yaml` |
| Understanding | AR only | 文本 + 图像 | 文本 | `mammoth_moda2_ar.yaml` |

## 3. 核心创新点

### 3.1 MoE (Mixture of Experts) 架构
- **Understanding Expert (mlp)**: 处理文本理解任务
- **Generation Expert (gen_mlp)**: 处理图像生成任务
- 通过 `gen_token_mask` 动态路由到不同的专家网络

### 3.2 双词汇表设计
- **Base Vocabulary**: 标准文本 token (vocab_size ~152064)
- **Generation Vocabulary**: 图像生成 token (gen_vocab_size ~32800)
- 总词汇量: ~184864 tokens

### 3.3 实值旋转位置编码 (Real-valued RoPE)
- 支持 3D 位置编码 (时间, 高度, 宽度)
- 用于 DiT 阶段的图像 token 位置表示

## 4. 关键文件索引

| 组件 | 文件路径 |
|------|----------|
| 主模型入口 | `vllm_omni/model_executor/models/mammoth_moda2/mammoth_moda2.py` |
| 配置类 | `vllm_omni/transformers_utils/configs/mammoth_moda2.py` |
| DiT 模型 | `vllm_omni/diffusion/models/mammoth_moda2/mammothmoda2_dit_model.py` |
| DiT Pipeline | `vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py` |
| Stage 转换 | `vllm_omni/model_executor/stage_input_processors/mammoth_moda2.py` |
| Tokenizer | `vllm_omni/tokenizers/mammoth_moda2_tokenizer.py` |
| RoPE | `vllm_omni/diffusion/models/mammoth_moda2/rope_real.py` |
| Scheduler | `vllm_omni/diffusion/models/mammoth_moda2/schedulers.py` |
