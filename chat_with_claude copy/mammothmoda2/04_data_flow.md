# 数据流向与传播过程

## 1. T2I (Text-to-Image) 完整流程

### 1.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        T2I 生成完整流程                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  用户输入: "画一只猫"                                                         │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Tokenizer (MammothUTokenizer)                     │   │
│  │  "画一只猫" → [151644, 894739, 894740, ...]                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    AR Stage (MammothModa2ARForConditionalGeneration)│   │
│  │                                                                       │   │
│  │  Step 1: embed_tokens(input_ids)                                     │   │
│  │          [batch, seq_len] → [batch, seq_len, 8192]                   │   │
│  │                                                                       │   │
│  │  Step 2: 80 × Mammoth2DecoderLayer                                   │   │
│  │          每个 token 根据 gen_token_mask 路由到:                        │   │
│  │          - mlp (文本 token)                                          │   │
│  │          - gen_mlp (图像 token)                                      │   │
│  │                                                                       │   │
│  │  Step 3: 自回归生成 visual tokens                                     │   │
│  │          每步采样一个 token, 直到生成完整 AR grid                      │   │
│  │          AR grid 大小: ar_width × ar_height                          │   │
│  │                                                                       │   │
│  │  输出:                                                                │   │
│  │    - text_hidden_states: [total_seq_len, 8192]                      │   │
│  │    - generated_token_ids: [ar_width * (ar_height + 1)]              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ar2dit Processor                                  │   │
│  │                                                                       │   │
│  │  1. 分离 hidden states:                                              │   │
│  │     - text_condition: prompt 部分的文本 hidden states                │   │
│  │     - image_condition: 生成的 visual token hidden states             │   │
│  │                                                                       │   │
│  │  2. 构建 additional_information:                                     │   │
│  │     - text_prompt_embeds: [T_text, 8192]                            │   │
│  │     - image_prompt_embeds: [T_img, 8192]                            │   │
│  │     - image_height, image_width                                      │   │
│  │     - text_guidance_scale, cfg_range                                 │   │
│  │     - num_inference_steps                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DiT Stage (MammothModa2DiTPipeline)              │   │
│  │                                                                       │   │
│  │  Step 1: 条件准备                                                    │   │
│  │          text_embeds = caption_embedder(text_prompt_embeds)          │   │
│  │          image_embeds = image_prompt_embeds (可选 refiner)           │   │
│  │          prompt_embeds = concat(text_embeds, image_embeds)           │   │
│  │                                                                       │   │
│  │  Step 2: 初始化噪声                                                   │   │
│  │          latents = randn([1, 16, H/8, W/8])                          │   │
│  │                                                                       │   │
│  │  Step 3: 扩散循环 (num_inference_steps 步)                           │   │
│  │          for t in timesteps:                                         │   │
│  │              # DiT 前向                                              │   │
│  │              pred = gen_transformer(latents, t, prompt_embeds)       │   │
│  │                                                                       │   │
│  │              # CFG (可选)                                            │   │
│  │              if guidance_scale > 1:                                  │   │
│  │                  pred_uncond = gen_transformer(latents, t, null)     │   │
│  │                  pred = uncond + scale * (pred - uncond)             │   │
│  │                                                                       │   │
│  │              # Euler 步进                                            │   │
│  │              latents = latents + (t_next - t) * pred                 │   │
│  │                                                                       │   │
│  │  Step 4: VAE 解码                                                    │   │
│  │          image = gen_vae.decode(latents)                             │   │
│  │          image: [1, 3, H, W]                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  输出图像: [1, 3, H, W]                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 AR 阶段自回归生成详解

```
初始状态:
  prompt_token_ids:     [151644, 894739, 894740, ...] (用户输入)
  generated_token_ids:  [] (空)

生成循环:

Iteration 0:
  input_ids = prompt_token_ids
              │
              ▼ forward
  hidden_states: [len(prompt), hidden_size]
              │
              ▼ 取最后一个 token 的 hidden state
  last_hidden: [hidden_size]
              │
              ▼ lm_head + gen_head
  logits: [total_vocab_size]
              │
              ▼ _apply_t2i_token_constraints (第一个位置)
  constrained_logits: 只有 visual tokens 非 -inf
              │
              ▼ sample
  next_token: visual_token_id (如 152064)
  
Iteration 1:
  input_ids = prompt_token_ids + [visual_token_id]
              │
              ▼ forward
  ... (同上)
  
  第 2 个 token 约束:
    - 如果 column_id == ar_width: 只允许 eol_token_id
    - 否则: 只允许 visual_token 范围

Iteration N (生成 ar_width * ar_height 个 token):
  最终生成序列: [visual_1, visual_2, ..., eol, visual_3, ..., eol, ...]
               共 ar_width * (ar_height + 1) 个 token (每行末尾有 eol)
```

### 1.3 AR Grid 结构

```
AR Grid 输出格式 (ar_width × ar_height):

Token 位置布局:
┌─────────────────────────────────────────────────────────────┐
│  [v1] [v2] [v3] ... [v_ar_width] [EOL]     ← Row 0         │
│  [v_ar_width+2] ... [v_2*ar_width+1] [EOL] ← Row 1         │
│  ...                                                        │
│  [v_N-ar_width] ... [v_N-1] [EOL]           ← Row ar_height│
└─────────────────────────────────────────────────────────────┘

其中:
  - v_i: visual token ID (范围: [gen_vocab_start_index, gen_vocab_start_index + gen_vocab_size))
  - EOL: end-of-line token ID
  - 总 token 数: ar_width × (ar_height + 1)
```

---

## 2. Understanding 模式流程

### 2.1 图像理解流程

```
用户输入: 图像 + "描述这张图片"
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    多模态输入处理                                     │
│                                                                       │
│  1. 图像预处理:                                                       │
│     image → ViT encoder → vision_hidden_states                       │
│     [H, W, 3] → [num_patches, hidden_size]                          │
│                                                                       │
│  2. 文本 tokenization:                                               │
│     "描述这张图片" → [token_ids]                                      │
│                                                                       │
│  3. 构建输入序列:                                                     │
│     [vision_start] + [vision_tokens] + [vision_end] + [text_tokens] │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AR Stage 推理                                     │
│                                                                       │
│  1. 嵌入查找:                                                         │
│     vision_tokens → visual.embed_tokens (Vision Tower)               │
│     text_tokens → embed_tokens                                        │
│                                                                       │
│  2. 全部 token 都走 mlp (无 gen_token)                               │
│                                                                       │
│  3. 自回归生成文本回答                                                │
│     logits 中 gen vocab 部分被 mask 为 -inf                          │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
输出: 文本描述
```

---

## 3. ar2dit Processor 详解

**位置**: `stage_input_processors/mammoth_moda2.py:11-100`

### 3.1 输入

```python
ar_output.outputs[0]:
  - prompt_token_ids: List[int]          # 用户输入 token IDs
  - cumulative_token_ids: List[int]      # 生成的 token IDs
  - multimodal_output["latent"]: Tensor  # 所有 token 的 hidden states

prompt.additional_information:
  - image_height: int
  - image_width: int
  - text_guidance_scale: float
  - cfg_range: Tuple[float, float]
  - num_inference_steps: int
  - visual_token_start_id: int
  - visual_ids: List[int]  # vision 相关 token IDs
```

### 3.2 处理逻辑

```python
def ar2dit(ar_outputs, prompts):
    for ar_output, prompt in zip(ar_outputs, prompts):
        # 1. 获取完整 token 序列
        prompt_token_ids = ar_output.prompt_token_ids
        gen_token_ids = ar_output.outputs[0].cumulative_token_ids[:-1]
        full_token_ids = prompt_token_ids + gen_token_ids
        
        # 2. 获取完整 hidden states
        full_hidden_states = ar_output.multimodal_output["latent"]
        # full_hidden_states: [total_len, hidden_size]
        
        # 3. 构建 mask
        answer_start_index = len(prompt_token_ids)
        
        # 问题部分 mask
        questions_mask = pos < answer_start_index
        
        # 生成部分 mask
        answers_mask = ~questions_mask
        
        # visual token mask (来自 ViT 输出)
        visual_token_mask = token_ids in visual_ids
        
        # gen token mask (来自 AR 生成)
        gen_token_mask = token_ids >= gen_vocab_start_index
        
        # 4. 提取条件
        # 文本条件: 问题部分的非 visual 非 gen token
        text_condition_mask = questions_mask & ~visual_token_mask & ~gen_token_mask
        
        # 图像条件: 答案部分的 gen token
        image_condition_mask = answers_mask & gen_token_mask
        
        text_prompt_embeds = full_hidden_states[text_condition_mask]
        image_prompt_embeds = full_hidden_states[image_condition_mask]
        
        # 5. 构建 DiT 输入
        dit_input = OmniTokensPrompt(
            prompt_token_ids=[0],
            additional_information={
                "text_prompt_embeds": text_prompt_embeds,
                "image_prompt_embeds": image_prompt_embeds,
                "image_height": image_height,
                "image_width": image_width,
                "text_guidance_scale": text_guidance_scale,
                "cfg_range": cfg_range,
                "num_inference_steps": num_inference_steps,
            }
        )
```

### 3.3 输出

```python
OmniTokensPrompt:
  prompt_token_ids: [0]  # 占位符
  additional_information:
    text_prompt_embeds: [T_text, llm_hidden_size]       # 文本条件
    image_prompt_embeds: [T_img, llm_hidden_size]       # 图像条件
    image_height: int
    image_width: int
    text_guidance_scale: float
    cfg_range: [start, end]
    num_inference_steps: int
```

---

## 4. 张量形状汇总

### 4.1 AR 阶段

| 阶段 | 张量 | 形状 |
|------|------|------|
| 输入 | input_ids | [batch, seq_len] |
| 嵌入 | hidden_states | [batch, seq_len, 8192] |
| Attention | Q/K/V | [batch, seq_len, 64, 128] |
| FFN 输入 | hidden_states | [batch, seq_len, 8192] |
| FFN 中间 | intermediate | [batch, seq_len, 29568] |
| FFN 输出 | output | [batch, seq_len, 8192] |
| Logits | logits | [batch, seq_len, 184864] |

### 4.2 DiT 阶段

| 阶段 | 张量 | 形状 |
|------|------|------|
| 输入 | text_cond | [T_text, 8192] |
| 输入 | image_cond | [T_img, 8192] |
| 嵌入后 | prompt_embeds | [1, T_text+T_img, 2304] |
| 噪声 latent | latents | [1, 16, H/8, W/8] |
| Patch 嵌入 | img_tokens | [1, num_patches, 2304] |
| 联合序列 | joint_hidden | [1, T_text+num_patches, 2304] |
| DiT 输出 | output | [1, 16, H/8, W/8] |
| VAE 解码 | image | [1, 3, H, W] |

### 4.3 形状计算示例

假设:
- 输入文本: "画一只猫" (10 tokens)
- AR 生成: 32 × 32 grid = 1056 visual tokens + 32 EOL = 1088 tokens
- 输出图像: 512 × 512

```
AR 阶段:
  输入序列: 10 tokens
  生成序列: 10 + 1088 = 1098 tokens
  hidden_states: [1, 1098, 8192]
  logits: [1, 1098, 184864]

ar2dit:
  text_condition: ~10 tokens (prompt 部分)
  image_condition: 1056 tokens (visual tokens, 不含 EOL)
  text_prompt_embeds: [10, 8192]
  image_prompt_embeds: [1056, 8192]

DiT 阶段:
  prompt_embeds: [1, 10+1056, 2304]
  latents: [1, 16, 64, 64] (512/8 = 64)
  num_patches: 64 × 64 / 4 = 1024 (patch_size=2)
  img_tokens: [1, 1024, 2304]
  joint_hidden: [1, 1070+1024, 2304] = [1, 2094, 2304]
  
  扩散 N 步后:
  image: [1, 3, 512, 512]
```

---

## 5. 时间线与调度

### 5.1 AR 阶段时间消耗

```
每步生成:
  - Forward: ~50ms (80 layers, 8B params)
  - Sampling: ~1ms
  - 总共 ~1088 steps × 50ms = ~54s (无优化)
  
优化后 (使用 KV cache, speculative decoding 等):
  - 可优化至 ~5-10s
```

### 5.2 DiT 阶段时间消耗

```
默认设置 (num_inference_steps=20):
  - 每步 DiT forward: ~100ms (26 layers, hidden_size=2304)
  - CFG 开启时: ×2 (条件 + 无条件)
  - 总共: 20 × 100ms × 2 = 4s (CFG)
  - VAE decode: ~200ms
  
总计 DiT: ~4.2s
```

### 5.3 总体延迟

```
T2I 总延迟 ≈ AR 时间 + DiT 时间
           ≈ 5-10s + 4-5s
           ≈ 10-15s (优化后)
```
