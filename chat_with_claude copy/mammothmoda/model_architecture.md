# MammothModa2 模型架构详解

## 目录

1. [概述](#1-概述)
2. [模型架构](#2-模型架构)
3. [AR Stage (自回归阶段)](#3-ar-stage-自回归阶段)
4. [DiT Stage (扩散变换器阶段)](#4-dit-stage-扩散变换器阶段)
5. [端到端生成流程](#5-端到端生成流程)
6. [关键创新点](#6-关键创新点)
7. [代码结构](#7-代码结构)

---

## 1. 概述

MammothModa2 是一个**统一多模态理解与生成**的大模型，采用 **AR + DiT 双阶段架构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MammothModa2 整体架构                                │
│                                                                              │
│   输入: 文本 + 可选图像                                                       │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    AR Stage (自回归阶段)                             │   │
│   │                                                                      │   │
│   │   Vision Encoder ──► LLM Backbone (MoE) ──► AR Grid Tokens          │   │
│   │                                                                      │   │
│   │   功能: 多模态理解 + 潜在表示生成                                      │   │
│   │   输出: latent hidden states                                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              │ text_condition + image_condition             │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    DiT Stage (扩散变换器阶段)                        │   │
│   │                                                                      │   │
│   │   Condition Embeds ──► DiT Transformer ──► VAE Decoder ──► Image    │   │
│   │                                                                      │   │
│   │   功能: 潜在表示解码为图像                                            │   │
│   │   输出: 生成的图像                                                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 设计理念

MammothModa2 的核心思想是将**多模态理解**和**图像生成**统一在同一个模型框架内：

| 任务 | AR Stage | DiT Stage |
|------|----------|-----------|
| 文本理解 | ✅ 处理 | - |
| 图像理解 | ✅ 编码 | - |
| 文生图 (T2I) | ✅ 生成潜在表示 | ✅ 解码为图像 |
| 图生图 (I2I) | ✅ 理解+生成 | ✅ 解码为图像 |

### 1.2 模型规模

| 组件 | 参数规模 | 配置 |
|------|----------|------|
| LLM Backbone | ~72B | hidden_size=8192, layers=80, heads=64 |
| Vision Encoder | ~600M | 基于 Qwen2.5-VL |
| DiT Transformer | ~2B | 扩散变换器 |
| VAE Decoder | ~80M | AutoencoderKL |

---

## 2. 模型架构

### 2.1 配置层级

```
Mammothmoda2Config (顶层配置)
│
├── llm_config: Mammothmoda2Qwen2_5_VLConfig
│   │
│   ├── text_config: Mammothmoda2Qwen2_5_VLTextConfig
│   │   ├── hidden_size: 8192
│   │   ├── intermediate_size: 29568
│   │   ├── num_hidden_layers: 80
│   │   ├── num_attention_heads: 64
│   │   ├── num_key_value_heads: 8 (GQA)
│   │   ├── vocab_size: 152064
│   │   ├── extra_gen_vocab: True
│   │   └── gen_vocab_size: 32800
│   │
│   └── vision_config: Mammothmoda2Qwen2_5_VLVisionConfig
│       ├── hidden_size: 3584
│       ├── depth: 32
│       └── patch_size: 14
│
├── gen_vae_config: dict (VAE 配置)
├── gen_dit_config: dict (DiT 配置)
├── gen_axes_dim_rope: [40, 40, 40]
├── gen_axes_lens: [10000, 10000, 10000]
└── gen_condition_mode: "image"
```

### 2.2 词汇表结构

MammothModa2 使用**扩展词汇表**来支持图像生成：

```
词汇表结构:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Token ID Range         │ 用途                    │ 大小                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  0 - 152063             │ 基础文本词汇             │ 152064                │
│  152064 - 184863        │ 图像生成专用词汇         │ 32800 (gen_vocab)     │
│                         │ (AR Grid Tokens)        │                       │
└─────────────────────────────────────────────────────────────────────────────┘

AR Grid Token 示例:
  - 每个 token 代表图像 latent 空间的一个位置
  - 按 grid 顺序排列: [row0_col0, row0_col1, ..., row0_eol, row1_col0, ...]
  - EOL token 标记每行结束
```

---

## 3. AR Stage (自回归阶段)

### 3.1 组件结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MammothModa2ARForConditionalGeneration                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Vision Encoder (继承自 Qwen2.5-VL)              │   │
│  │                                                                      │   │
│  │   输入: 图像像素                                                      │   │
│  │   处理: Patch Embedding + Vision Transformer (32层)                  │   │
│  │   输出: 视觉特征序列 [batch, num_patches, 3584]                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 Language Model (MammothModa2Qwen2ForCausalLM)        │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                    Embeddings                                │   │   │
│  │   │                                                              │   │   │
│  │   │   embed_tokens: 文本嵌入 (vocab_size=152064, hidden=8192)    │   │   │
│  │   │   gen_embed_tokens: 生成嵌入 (vocab_size=32800, hidden=8192) │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │              80x Mammoth2DecoderLayer                        │   │   │
│  │   │                                                              │   │   │
│  │   │   每层结构:                                                   │   │   │
│  │   │   ┌────────────────────────────────────────────────────────┐ │   │   │
│  │   │   │  Input LayerNorm                                       │ │   │   │
│  │   │   │         │                                               │ │   │   │
│  │   │   │         ▼                                               │ │   │   │
│  │   │   │  Self-Attention (GQA: 64 heads, 8 KV heads)            │ │   │   │
│  │   │   │         │                                               │ │   │   │
│  │   │   │         ▼                                               │ │   │   │
│  │   │   │  Post-Attention LayerNorm                               │ │   │   │
│  │   │   │         │                                               │ │   │   │
│  │   │   │         ▼                                               │ │   │   │
│  │   │   │  MoE MLP (Mixture of Experts)                          │ │   │   │
│  │   │   │    ├── und_mlp: 理解专家 (文本 token 使用)              │ │   │   │
│  │   │   │    └── gen_mlp: 生成专家 (图像生成 token 使用)           │ │   │   │
│  │   │   └────────────────────────────────────────────────────────┘ │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                    Output Heads                              │   │   │
│  │   │                                                              │   │   │
│  │   │   lm_head: 文本预测头 (vocab_size=152064)                    │   │   │
│  │   │   gen_head: 生成预测头 (vocab_size=32800)                    │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 MoE (Mixture of Experts) 机制

MammothModa2 使用 MoE 来区分**理解**和**生成**任务：

```python
def moe_forward(hidden_states, und_expert, gen_expert, gen_token_mask):
    """
    MoE 路由逻辑:
    
    gen_token_mask: True = 图像生成 token, 使用 gen_expert
                    False = 文本理解 token, 使用 und_expert
    """
    if gen_expert is None or gen_token_mask is None:
        return und_expert(hidden_states)  # 所有 token 走理解专家
    
    if not gen_token_mask.any():
        return und_expert(hidden_states)  # 所有 token 走理解专家
    
    if gen_token_mask.all():
        return gen_expert(hidden_states)  # 所有 token 走生成专家
    
    # 混合情况: 分别处理
    und_tokens = hidden_states[~gen_token_mask]
    gen_tokens = hidden_states[gen_token_mask]
    
    und_output = und_expert(und_tokens)
    gen_output = gen_expert(gen_tokens)
    
    # 合并并恢复原始顺序
    return merge_and_reorder(und_output, gen_output, gen_token_mask)
```

### 3.3 AR Grid 生成约束

在 T2I 任务中，AR 阶段需要按特定格式生成图像 latent 的 token 序列：

```
AR Grid 结构:

图像 latent 尺寸: [H, W]
Grid 尺寸: [H x (W+1)]  # 每个 row 后加一个 EOL token

Token 序列示例 (4x4 latent):
  [v0, v1, v2, v3, EOL, v4, v5, v6, v7, EOL, v8, v9, v10, v11, EOL, v12, v13, v14, v15, EOL]

约束逻辑:
  - 行内位置 (column < width): 只允许 visual token
  - 行末位置 (column == width): 只允许 EOL token
```

---

## 4. DiT Stage (扩散变换器阶段)

### 4.1 组件结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MammothModa2DiTPipeline                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Caption Embedder (重初始化)                       │   │
│  │                                                                      │   │
│  │   输入: text_condition + image_condition (来自 AR 阶段)              │   │
│  │   处理: Qwen2RMSNorm + Linear                                        │   │
│  │   输出: 条件嵌入 [batch, seq_len, dit_hidden_size]                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Image Condition Refiner (可选 Q-Former)                 │   │
│  │                                                                      │   │
│  │   功能: 细化图像条件嵌入                                              │   │
│  │   输入: image_condition embeds                                       │   │
│  │   输出: refined image embeds                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DiT Transformer                                 │   │
│  │                                                                      │   │
│  │   架构: 基于 Flow Matching 的扩散变换器                               │   │
│  │                                                                      │   │
│  │   输入:                                                              │   │
│  │     - latents: 噪声潜在表示 [batch, C, H, W]                         │   │
│  │     - timestep: 扩散时间步                                           │   │
│  │     - prompt_embeds: 条件嵌入                                        │   │
│  │                                                                      │   │
│  │   处理:                                                              │   │
│  │     - Patchify latents → token sequence                              │   │
│  │     - RoPE 位置编码 (3D: height, width, time)                        │   │
│  │     - Transformer blocks (cross-attention on prompt)                 │   │
│  │     - Unpatchify → latents                                           │   │
│  │                                                                      │   │
│  │   输出: 去噪预测                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         VAE Decoder                                  │   │
│  │                                                                      │   │
│  │   输入: latents [batch, C, H/16, W/16]                               │   │
│  │   处理: 反卷积 + 上采样                                               │   │
│  │   输出: 图像 [batch, 3, H, W]                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 扩散采样流程

```
Flow Matching Euler 采样:

初始化:
  latents ~ N(0, I)  # 纯噪声

扩散循环 (num_inference_steps 步):
  
  for t in timesteps:
      # 条件预测
      model_pred = DiT(latents, t, prompt_embeds)
      
      # CFG (Classifier-Free Guidance)
      if guidance_scale > 1.0:
          model_pred_uncond = DiT(latents, t, negative_embeds)
          model_pred = uncond + scale * (cond - uncond)
      
      # Euler 步进
      latents = scheduler.step(model_pred, t, latents)
  
VAE 解码:
  latents = latents / scaling_factor + shift_factor
  image = VAE.decode(latents)
```

### 4.3 CFG (Classifier-Free Guidance)

```
CFG 公式:

  pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

参数:
  - guidance_scale: 控制生成结果与条件的对齐程度
  - cfg_range: [start, end] 控制在哪些步应用 CFG

示例:
  - guidance_scale = 4.0: 较强的条件引导
  - cfg_range = [0.0, 1.0]: 全程应用 CFG
  - cfg_range = [0.5, 1.0]: 后半程应用 CFG
```

---

## 5. 端到端生成流程

### 5.1 T2I (文本生成图像) 流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户输入                                        │
│                                                                              │
│   prompt: "A stylish woman riding a motorcycle in NYC, movie poster style"  │
│   height: 1024, width: 1024                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           预处理阶段                                         │
│                                                                              │
│   1. Tokenize prompt → token_ids                                            │
│   2. 计算 AR grid 尺寸: grid_h = height/16, grid_w = width/16               │
│   3. 准备生成参数: guidance_scale, num_steps, etc.                          │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AR Stage 处理                                       │
│                                                                              │
│   输入: prompt token_ids                                                    │
│                                                                              │
│   自回归生成循环:                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  for step in range(num_ar_tokens):                                  │   │
│   │      # Forward pass                                                 │   │
│   │      hidden_states = LLM(input_ids, positions)                      │   │
│   │                                                                     │   │
│   │      # Compute logits                                               │   │
│   │      base_logits = lm_head(hidden_states)                           │   │
│   │      gen_logits = gen_head(hidden_states)                           │   │
│   │      logits = concat([base_logits, gen_logits])                     │   │
│   │                                                                     │   │
│   │      # Apply AR grid constraints                                    │   │
│   │      if is_end_of_row:                                              │   │
│   │          logits = mask_all_except(eol_token)                        │   │
│   │      else:                                                          │   │
│   │          logits = mask_non_visual_tokens()                          │   │
│   │                                                                     │   │
│   │      # Sample next token                                            │   │
│   │      next_token = sample(logits)                                    │   │
│   │      input_ids = append(input_ids, next_token)                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   输出:                                                                      │
│     - generated_token_ids: AR grid tokens                                   │
│     - hidden_states: 所有 token 的隐藏状态                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ar2dit 转换器                                        │
│                                                                              │
│   功能: 提取条件嵌入                                                         │
│                                                                              │
│   处理:                                                                      │
│   1. 分离 prompt token 和 generated token                                   │
│   2. 根据 token mask 提取:                                                   │
│      - text_condition: prompt 中的文本 token 隐藏状态                        │
│      - image_condition: generated 中的视觉 token 隐藏状态                    │
│   3. 转换为 float32                                                          │
│                                                                              │
│   输出:                                                                      │
│     - text_prompt_embeds: [num_text_tokens, hidden_size]                    │
│     - image_prompt_embeds: [num_visual_tokens, hidden_size]                 │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DiT Stage 处理                                       │
│                                                                              │
│   输入:                                                                      │
│     - text_prompt_embeds                                                    │
│     - image_prompt_embeds                                                   │
│     - height, width, guidance_scale, num_steps                              │
│                                                                              │
│   处理:                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  1. 合并条件嵌入: prompt_embeds = concat([text, image])              │   │
│   │                                                                      │   │
│   │  2. 初始化噪声: latents = randn([1, C, H/8, W/8])                    │   │
│   │                                                                      │   │
│   │  3. 扩散采样循环:                                                     │   │
│   │     for t in timesteps:                                              │   │
│   │         pred = DiT(latents, t, prompt_embeds)                        │   │
│   │         if use_cfg:                                                  │   │
│   │             pred_uncond = DiT(latents, t, negative_embeds)           │   │
│   │             pred = cfg_combine(pred, pred_uncond, scale)             │   │
│   │         latents = scheduler.step(pred, t, latents)                   │   │
│   │                                                                      │   │
│   │  4. VAE 解码: image = VAE.decode(latents)                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   输出: image [1, 3, height, width]                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              最终输出                                        │
│                                                                              │
│   output.png: 1024x1024 RGB 图像                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 I2I (图像生成图像) 流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户输入                                        │
│                                                                              │
│   prompt: "Transform to watercolor style"                                   │
│   input_image: PIL.Image                                                    │
│   height: 1024, width: 1024                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AR Stage 处理                                       │
│                                                                              │
│   与 T2I 类似，但:                                                           │
│   1. 输入包含 input_image                                                    │
│   2. Vision Encoder 编码 input_image                                        │
│   3. LLM 同时处理文本和图像特征                                               │
│   4. 生成过程中，image_condition 来自输入图像的理解表示                       │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         DiT Stage (同 T2I)
                              │
                              ▼
                         输出图像
```

---

## 6. 关键创新点

### 6.1 统一的理解-生成架构

```
传统方案:                          MammothModa2:

┌─────────────────┐               ┌─────────────────────────────┐
│  理解模型 (LLM)  │               │                             │
│  - 独立训练      │               │   统一模型                   │
│  - 独立推理      │               │   - 共享表示空间             │
└─────────────────┘               │   - 单次前向传播             │
                                  │   - 理解 + 生成无缝衔接      │
┌─────────────────┐               │                             │
│  生成模型 (DiT)  │               └─────────────────────────────┘
│  - 独立训练      │
│  - 独立推理      │
│  - 需要 adapter │
└─────────────────┘
```

### 6.2 MoE 专家分离

```
理解专家 (und_mlp):              生成专家 (gen_mlp):

┌─────────────────────┐          ┌─────────────────────┐
│ 文本理解任务         │          │ 图像生成任务         │
│ - 问答              │          │ - AR grid 生成       │
│ - 摘要              │          │ - 潜在表示编码       │
│ - 翻译              │          │                     │
│                     │          │ 特点:               │
│ 特点:               │          │ - 学习视觉 token 分布 │
│ - 保持语言能力      │          │ - 与 DiT 阶段对齐    │
│ - 不受生成干扰      │          │                     │
└─────────────────────┘          └─────────────────────┘

优势:
  - 解耦理解与生成，避免任务冲突
  - 各专家专注优化，性能更优
  - 推理时按需激活，效率更高
```

### 6.3 AR Grid Token 表示

```
传统 VQ-GAN/DALL-E:              MammothModa2 AR Grid:

┌─────────────────────┐          ┌─────────────────────┐
│ 离散编码本           │          │ 连续潜在表示         │
│ - 学习 codebook     │          │ - 无需离散化        │
│ - 可能丢失细节      │          │ - 保留更多细节      │
│                     │          │                     │
│ [c1, c2, c3, ...]   │          │ [h1, h2, h3, ...]   │
│ (离散 code)         │          │ (连续 hidden state) │
└─────────────────────┘          └─────────────────────┘
        │                                │
        ▼                                ▼
   解码器解码                      直接用于 DiT
   (信息瓶颈)                     (无信息损失)
```

---

## 7. 代码结构

### 7.1 文件组织

```
vllm_omni/
├── transformers_utils/configs/mammoth_moda2.py
│   └── Mammothmoda2Config, Mammothmoda2Qwen2_5_VLConfig, ...
│
├── model_executor/models/mammoth_moda2/
│   ├── mammoth_moda2.py
│   │   ├── MammothModa2ForConditionalGeneration (顶层模型)
│   │   ├── MammothModa2ARForConditionalGeneration (AR 阶段)
│   │   ├── MammothModa2Qwen2ForCausalLM (语言模型)
│   │   ├── Mammoth2DecoderLayer (MoE 层)
│   │   └── moe_forward, moe_enable (MoE 工具函数)
│   │
│   └── stage_configs/
│       ├── mammoth_moda2.yaml (完整 T2I 流程)
│       └── mammoth_moda2_ar.yaml (仅理解任务)
│
├── diffusion/models/mammoth_moda2/
│   ├── pipeline_mammothmoda2_dit.py
│   │   └── MammothModa2DiTPipeline (DiT 阶段)
│   ├── mammothmoda2_dit_model.py
│   │   └── Transformer2DModel, SimpleQFormerImageRefiner
│   ├── rope_real.py
│   │   └── RotaryPosEmbedReal (3D RoPE)
│   └── schedulers/
│       └── FlowMatchEulerDiscreteScheduler
│
├── model_executor/stage_input_processors/mammoth_moda2.py
│   └── ar2dit (AR → DiT 数据转换)
│
└── worker/
    ├── gpu_ar_worker.py (AR Worker)
    └── gpu_generation_worker.py (DiT Worker)
```

### 7.2 类继承关系

```
PretrainedConfig
    └── Mammothmoda2Config
            └── Mammothmoda2Qwen2_5_VLConfig
                    ├── Mammothmoda2Qwen2_5_VLTextConfig
                    └── Mammothmoda2Qwen2_5_VLVisionConfig

nn.Module
    ├── MammothModa2ForConditionalGeneration (顶层入口)
    │       ├── MammothModa2ARForConditionalGeneration (AR)
    │       │       └── Qwen2_5_VLForConditionalGeneration
    │       │               └── MammothModa2Qwen2ForCausalLM
    │       │                       └── Mammoth2DecoderLayer
    │       │
    │       └── MammothModa2DiTPipeline (DiT)
    │               ├── Transformer2DModel
    │               └── AutoencoderKL
    │
    └── Mammoth2DecoderLayer
            └── Qwen2DecoderLayer
```

---

## 附录：推理示例

### CLI 调用

```bash
python examples/offline_inference/mammothmodal2_preview/run_mammothmoda2_t2i.py \
  --model bytedance-research/MammothModa2-Preview \
  --stage-config ./vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 50 \
  --text-guidance-scale 4.0 \
  --out output.png
```

### Python API

```python
from vllm_omni import Omni

omni = Omni(
    model="bytedance-research/MammothModa2-Preview",
    stage_configs_path="./vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml",
)

outputs = omni.generate(
    prompts=["A cat sitting on a tree branch"],
    height=1024,
    width=1024,
    num_inference_steps=50,
    text_guidance_scale=4.0,
)

outputs[0].save("output.png")
```
