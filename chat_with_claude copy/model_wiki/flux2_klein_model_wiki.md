# Flux2-Klein 多模态扩散模型详解

## 目录

1. [模型概述](#1-模型概述)
2. [核心组件架构](#2-核心组件架构)
3. [推理流程详解](#3-推理流程详解)
4. [关键组件生命周期](#4-关键组件生命周期)
5. [核心机制解析](#5-核心机制解析)
6. [vLLM-Omni优化技术](#6-vllm-omni优化技术)
7. [矩阵形状示例](#7-矩阵形状示例)

---

## 1. 模型概述

Flux2-Klein 是 Black Forest Labs 开发的文本到图像生成扩散模型，基于 Flow Matching（流匹配）范式。该模型属于 Transformer-based Diffusion Model（基于 Transformer 的扩散模型），与传统的 UNet 架构不同。

### 1.1 模型特点

| 特性 | 描述 |
|------|------|
| 架构类型 | DiT (Diffusion Transformer) |
| 扩散范式 | Flow Matching (Rectified Flow) |
| 文本编码器 | Qwen3ForCausalLM |
| 图像编解码器 | AutoencoderKLFlux2 |
| 位置编码 | 4D RoPE (Rotary Position Embedding) |
| 注意力机制 | 双流 + 单流混合架构 |

### 1.2 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      Flux2KleinPipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │  Text Input  │───▶│        Qwen3ForCausalLM              │  │
│  │   (Prompt)   │    │        (Text Encoder)                │  │
│  └──────────────┘    └──────────────┬───────────────────────┘  │
│                                     │ prompt_embeds             │
│                                     ▼                           │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │  Noise/Z_t   │───▶│                                      │  │
│  └──────────────┘    │     Flux2Transformer2DModel          │  │
│                      │  ┌─────────────────────────────────┐  │  │
│  ┌──────────────┐    │  │ Double-Stream Blocks (x N)     │  │  │
│  │ Timestep t   │───▶│  │   - Flux2TransformerBlock      │  │  │
│  └──────────────┘    │  │   - Joint Attention            │  │  │
│                      │  └─────────────────────────────────┘  │  │
│                      │  ┌─────────────────────────────────┐  │  │
│                      │  │ Single-Stream Blocks (x M)     │  │  │
│                      │  │   - Flux2SingleTransformerBlock │  │  │
│                      │  │   - Fused Attention + MLP       │  │  │
│                      │  └─────────────────────────────────┘  │  │
│                      └──────────────┬───────────────────────┘  │
│                                     │ noise_pred                │
│                                     ▼                           │
│                      ┌──────────────────────────────────────┐  │
│                      │   FlowMatchEulerDiscreteScheduler    │  │
│                      │        (Denoising Loop)              │  │
│                      └──────────────┬───────────────────────┘  │
│                                     │ latents                   │
│                                     ▼                           │
│                      ┌──────────────────────────────────────┐  │
│                      │        AutoencoderKLFlux2            │  │
│                      │          (VAE Decoder)               │  │
│                      └──────────────┬───────────────────────┘  │
│                                     │ image                     │
│                                     ▼                           │
│                      ┌──────────────────────────────────────┐  │
│                      │         Flux2ImageProcessor          │  │
│                      │        (Post-processing)             │  │
│                      └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心组件架构

### 2.1 Flux2Transformer2DModel（核心Transformer）

这是模型的主体，包含所有噪声预测逻辑。

```python
# 位置: flux2_klein_transformer.py
class Flux2Transformer2DModel(nn.Module):
    """
    核心架构组件:
    - pos_embed: 4D位置编码 (T, H, W, L)
    - time_guidance_embed: 时间步和引导嵌入
    - x_embedder: 图像token嵌入
    - context_embedder: 文本token嵌入
    - transformer_blocks: 双流注意力块
    - single_transformer_blocks: 单流注意力块
    """
```

#### 2.1.1 模型参数配置

```python
# 默认配置
config = {
    "patch_size": 1,
    "in_channels": 128,           # VAE latent channels * 4 (patchified)
    "num_layers": 8,              # Double-stream blocks数量
    "num_single_layers": 48,      # Single-stream blocks数量
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "joint_attention_dim": 15360, # 文本编码器输出维度
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),  # RoPE各维度
    "rope_theta": 2000,
}
```

### 2.2 Flux2KleinPipeline（推理管道）

管道协调所有组件的交互，负责端到端的图像生成。

```python
# 位置: pipeline_flux2_klein.py
class Flux2KleinPipeline(nn.Module, CFGParallelMixin, SupportImageInput):
    """
    核心组件:
    - text_encoder: Qwen3ForCausalLM
    - tokenizer: Qwen2TokenizerFast
    - transformer: Flux2Transformer2DModel
    - vae: AutoencoderKLFlux2
    - scheduler: FlowMatchEulerDiscreteScheduler
    - image_processor: Flux2ImageProcessor
    """
```

### 2.3 关键子组件详解

#### 2.3.1 Flux2Modulation（调制机制）

```python
class Flux2Modulation(nn.Module):
    """
    作用: 从时间嵌入生成调制参数 (shift, scale, gate)

    输入: temb [B, D] - 时间步嵌入
    输出: tuple of (shift, scale, gate) 参数组

    机制:
    - 通过SiLU激活 + Linear变换
    - 输出 3 * mod_param_sets 个参数
    - 用于AdaLN (Adaptive Layer Normalization)
    """

    def forward(self, temb):
        mod = self.act_fn(temb)  # SiLU
        mod = self.linear(mod)   # [B, D * 3 * mod_param_sets]
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        return tuple(mod_params[3*i : 3*(i+1)] for i in range(self.mod_param_sets))
```

**调制参数的作用:**

```
norm_hidden_states = (1 + scale) * norm_hidden_states + shift
hidden_states = hidden_states + gate * attn_output
```

#### 2.3.2 Flux2PosEmbed（位置编码）

```python
class Flux2PosEmbed(nn.Module):
    """
    作用: 计算4D旋转位置编码

    输入: ids [S, 4] - 位置坐标 (T, H, W, L)
    输出: (cos, sin) 旋转嵌入

    维度说明:
    - T: 时间维度 (帧)
    - H: 高度维度
    - W: 宽度维度
    - L: 层级/序列维度
    """
```

#### 2.3.3 Flux2Attention（注意力层）

```python
class Flux2Attention(nn.Module):
    """
    双流注意力机制

    特点:
    1. 支持图像和文本的联合注意力
    2. 使用RoPE位置编码
    3. 支持Sequence Parallel
    4. 使用RMSNorm归一化Q/K

    关键组件:
    - to_qkv: QKV并行投影 (用于图像tokens)
    - add_kv_proj: 额外的KV投影 (用于文本tokens)
    - rope: 旋转位置编码
    - attn: 通用注意力实现
    """
```

#### 2.3.4 Flux2ParallelSelfAttention（并行自注意力）

```python
class Flux2ParallelSelfAttention(nn.Module):
    """
    单流注意力 + MLP融合

    特点:
    1. QKV和MLP输入投影合并
    2. 减少内存访问次数
    3. 适用于Single-Stream Blocks
    """
```

---

## 3. 推理流程详解

### 3.1 完整推理流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Flux2-Klein 推理流程                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: 输入预处理                                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ prompt: str → tokenizer → input_ids [B, 512]               │   │
│  │ image: PIL.Image → resize → normalize → [B, 3, H, W]       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 2: 文本编码                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ input_ids → Qwen3ForCausalLM → hidden_states [layers 9,18,27]│   │
│  │ → stack → permute → prompt_embeds [B, 512, 15360]           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 3: 初始化潜在表示                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ randn_tensor → latents [B, 128*4, H/16, W/16]               │   │
│  │ → pack → latents [B, H*W/64, 128]                           │   │
│  │ → prepare latent_ids [B, H*W/64, 4]                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 4: 去噪循环                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ for t in timesteps:                                         │   │
│  │     ├─ 时间步嵌入 → temb                                    │   │
│  │     ├─ 调制参数 → (shift, scale, gate)                      │   │
│  │     ├─ Double-Stream Blocks (x8)                            │   │
│  │     │   └─ Joint Attention on (text, image)                 │   │
│  │     ├─ Single-Stream Blocks (x48)                           │   │
│  │     │   └─ Fused Self-Attention + MLP                       │   │
│  │     ├─ 输出投影 → noise_pred                                 │   │
│  │     └─ Scheduler step → latents_{t-1}                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 5: VAE解码                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ latents → unpack → unpatchify → [B, 128, H, W]              │   │
│  │ → VAE.decode → image [B, 3, H*16, W*16]                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Step 6: 后处理                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ image → denormalize → clip(0, 1) → PIL.Image                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 详细代码流程

#### Step 1: 输入检查与预处理

```python
# pipeline_flux2_klein.py: forward()

# 1.1 解析请求
prompt = req.prompts[0]  # 获取提示词
height = req.sampling_params.height
width = req.sampling_params.width
num_inference_steps = req.sampling_params.num_inference_steps
guidance_scale = req.sampling_params.guidance_scale

# 1.2 检查输入有效性
self.check_inputs(prompt, height, width, guidance_scale)
```

#### Step 2: 文本编码

```python
# 2.1 获取文本嵌入
prompt_embeds, text_ids = self.encode_prompt(
    prompt=prompt,
    device=device,
    max_sequence_length=512,
    text_encoder_out_layers=(9, 18, 27),  # 取中间层的隐藏状态
)

# 文本编码详细流程
def _get_qwen3_prompt_embeds(text_encoder, tokenizer, prompt):
    # 2.1.1 应用聊天模板
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)

    # 2.1.2 分词
    inputs = tokenizer(text, max_length=512, padding="max_length")
    input_ids = inputs["input_ids"]  # [B, 512]

    # 2.1.3 前向传播获取隐藏状态
    output = text_encoder(input_ids, output_hidden_states=True)

    # 2.1.4 取多层隐藏状态并堆叠
    # output.hidden_states: tuple of [B, 512, D] for each layer
    out = torch.stack([output.hidden_states[k] for k in (9, 18, 27)], dim=1)
    # out: [B, 3, 512, D]

    # 2.1.5 重塑为最终嵌入
    out = out.permute(0, 2, 1, 3).reshape(B, 512, 3*D)
    # prompt_embeds: [B, 512, 15360]

    return out
```

**矩阵形状示例:**
```
输入文本: "A beautiful sunset over the ocean"
↓ tokenizer
input_ids: [1, 512]  # batch_size=1, max_length=512
↓ Qwen3ForCausalLM
hidden_states[9]:  [1, 512, 5120]  # layer 9
hidden_states[18]: [1, 512, 5120]  # layer 18
hidden_states[27]: [1, 512, 5120]  # layer 27
↓ stack & reshape
prompt_embeds: [1, 512, 15360]  # 5120 * 3 = 15360
```

#### Step 3: 潜在表示初始化

```python
# 3.1 计算潜在空间尺寸
height = 2 * (int(height) // (self.vae_scale_factor * 2))
width = 2 * (int(width) // (self.vae_scale_factor * 2))
# vae_scale_factor = 16, 所以 height//32 * 2

# 3.2 生成噪声
shape = (batch_size, num_channels_latents * 4, height // 2, width // 2)
latents = randn_tensor(shape, generator=generator)
# latents: [B, 512, H/32, W/32] (假设128通道*4)

# 3.3 打包潜在表示
latents = self._pack_latents(latents)
# [B, C, H, W] -> [B, H*W, C]

# 3.4 准备位置ID
latent_ids = self._prepare_latent_ids(latents)
# latent_ids: [B, H*W, 4] 坐标 (T, H, W, L)
```

**Patchify操作详解:**
```python
def _patchify_latents(latents):
    """
    将2x2的patch压缩到channel维度

    输入: [B, C, H, W]
    输出: [B, C*4, H/2, W/2]

    例如: [1, 32, 64, 64] -> [1, 128, 32, 32]
    """
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H//2, 2, W//2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(B, C*4, H//2, W//2)
    return latents
```

#### Step 4: 去噪循环

```python
# 4.1 准备时间步
timesteps, num_inference_steps = retrieve_timesteps(
    self.scheduler, num_inference_steps, device
)

# 4.2 去噪循环
for i, t in enumerate(timesteps):
    # 4.2.1 时间步嵌入
    timestep = t.expand(latents.shape[0])

    # 4.2.2 Transformer前向传播
    noise_pred = self.transformer(
        hidden_states=latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timestep,
        img_ids=latent_ids,
        txt_ids=text_ids,
        guidance=guidance_scale,
    )

    # 4.2.3 CFG (Classifier-Free Guidance)
    if self.do_classifier_free_guidance:
        noise_pred_uncond = self.transformer(
            hidden_states=latents,
            encoder_hidden_states=negative_prompt_embeds,
            ...
        )
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

    # 4.2.4 调度器步骤
    latents = self.scheduler.step(noise_pred, t, latents)
```

#### Transformer内部流程

```python
# Flux2Transformer2DModel.forward()

def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
    # 1. 时间步嵌入
    temb = self.time_guidance_embed(timestep, guidance)
    # temb: [B, D] D = inner_dim (6144)

    # 2. 调制参数
    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)
    # 每个返回 (shift, scale, gate) 元组

    # 3. 输入嵌入
    hidden_states = self.x_embedder(hidden_states)
    # [B, seq_len, D]
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    # [B, txt_len, D]

    # 4. 位置编码
    txt_freqs_cos, txt_freqs_sin, img_freqs_cos, img_freqs_sin = \
        self.rope_prepare(img_ids, txt_ids)

    # 5. Double-Stream Blocks (图像和文本分离处理)
    for block in self.transformer_blocks:  # 8 blocks
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,          # 图像tokens
            encoder_hidden_states=encoder_hidden_states,  # 文本tokens
            temb_mod_params_img=double_stream_mod_img,
            temb_mod_params_txt=double_stream_mod_txt,
            image_rotary_emb=(freqs_cos, freqs_sin),
        )
        # 返回更新后的文本和图像tokens

    # 6. 合并文本和图像tokens
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # 7. Single-Stream Blocks (联合处理)
    for block in self.single_transformer_blocks:  # 48 blocks
        hidden_states = block(
            hidden_states=hidden_states,
            temb_mod_params=single_stream_mod,
            image_rotary_emb=(freqs_cos, freqs_sin),
            text_seq_len=num_txt_tokens,
        )

    # 8. 输出
    hidden_states = hidden_states[:, num_txt_tokens:]  # 移除文本tokens
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    return output
```

### 3.3 Flux2TransformerBlock详解

```python
class Flux2TransformerBlock(nn.Module):
    """
    双流Transformer块

    特点:
    1. 图像和文本分别处理
    2. 通过Joint Attention交互
    3. 各自有独立的FFN
    """

    def forward(self, hidden_states, encoder_hidden_states, temb_mod_params_img, temb_mod_params_txt):
        # 1. 调制参数解包
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        # 2. 图像分支 - 自注意力
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        # 3. 文本分支 - 自注意力
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        # 4. 联合注意力
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # 5. 残差连接
        hidden_states = hidden_states + gate_msa * attn_output
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output

        # 6. FFN
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_encoder_hidden_states)

        return encoder_hidden_states, hidden_states
```

### 3.4 Flux2SingleTransformerBlock详解

```python
class Flux2SingleTransformerBlock(nn.Module):
    """
    单流Transformer块

    特点:
    1. 图像和文本已合并为单一序列
    2. Attention和MLP融合
    3. 更高效的计算
    """

    def forward(self, hidden_states, temb_mod_params, text_seq_len):
        # 1. 调制参数
        mod_shift, mod_scale, mod_gate = temb_mod_params

        # 2. LayerNorm + 调制
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        # 3. 融合的Attention + MLP
        #    - QKV和MLP输入投影合并
        #    - 一次forward完成attention和MLP
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # 4. 残差连接
        hidden_states = hidden_states + mod_gate * attn_output

        return hidden_states
```

---

## 4. 关键组件生命周期

### 4.1 Pipeline生命周期

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline 生命周期                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                               │
│  │   初始化    │  __init__()                                   │
│  │   阶段     │  - 加载模型权重                                │
│  └──────┬──────┘  - 初始化组件                                 │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │   预热     │  (可选)                                        │
│  │   阶段     │  - 编译CUDA Graph                              │
│  └──────┬──────┘  - 预分配内存                                 │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │   推理     │  forward() [多次调用]                          │
│  │   阶段     │  - 处理请求                                    │
│  │            │  - 生成图像                                    │
│  └──────┬──────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐                                               │
│  │   销毁     │  (程序结束)                                    │
│  │   阶段     │  - 释放GPU内存                                 │
│  └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 各组件生命周期状态

| 组件 | 初始化时 | 推理时 | 推理后 |
|------|---------|--------|--------|
| TextEncoder | 加载权重 | 编码prompt | 保持状态(无KV缓存) |
| Transformer | 加载权重 | 噪声预测 | 保持状态 |
| VAE | 加载权重 | 编码/解码 | 保持状态 |
| Scheduler | 初始化参数 | 管理时间步 | 重置状态 |

### 4.3 权重加载流程

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
    """
    权重加载机制

    特点:
    1. 支持QKV融合权重的正确映射
    2. 支持量化权重
    3. 支持分布式加载
    """
    stacked_params_mapping = [
        (".to_qkv.", ".to_q.", "q"),
        (".to_qkv.", ".to_k.", "k"),
        (".to_qkv.", ".to_v.", "v"),
        # 额外的KV投影
        (".add_kv_proj", ".add_q_proj", "q"),
        (".add_kv_proj", ".add_k_proj", "k"),
        (".add_kv_proj", ".add_v_proj", "v"),
    ]

    for name, loaded_weight in weights:
        # 检查是否为融合参数
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name in name:
                # 映射到融合参数
                mapped_name = name.replace(weight_name, param_name)
                param = params_dict[mapped_name]
                weight_loader(param, loaded_weight, shard_id)
                break
```

---

## 5. 核心机制解析

### 5.1 Flow Matching调度器

Flux2使用Flow Matching（流匹配）而非传统的DDPM扩散。

```python
# FlowMatchEulerDiscreteScheduler

# 核心思想: 学习从噪声到数据的直线路径
# dx/dt = v(x_t, t) 其中 v 是速度场

# 前向过程 (添加噪声):
# x_1 = x_0 * (1 - t) + t * noise

# 反向过程 (去噪):
# x_{t-1} = x_t + dt * v_pred
```

**时间步调度:**
```python
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """
    计算最优的时间步分布参数mu

    基于图像序列长度和步数的经验公式
    优化去噪效率
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
    else:
        # 线性插值
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

    return mu
```

### 5.2 4D旋转位置编码

```python
class Flux2PosEmbed(nn.Module):
    """
    4D位置编码: (T, H, W, L)

    - T: 时间维度 (帧索引)
    - H: 高度维度
    - W: 宽度维度
    - L: 层级/序列维度

    每个维度使用独立的旋转角度
    """

    def forward(self, ids: torch.Tensor):
        """
        ids: [seq_len, 4] 位置坐标

        例如对于 32x32 的图像:
        - 位置 (0, 5, 10, 0) 表示:
          - T=0 (第一帧)
          - H=5 (第5行)
          - W=10 (第10列)
          - L=0 (层级0)
        """
        cos_out = []
        sin_out = []

        for i in range(len(self.axes_dim)):  # 4个维度
            freqs_cis = get_1d_rotary_pos_embed(
                self.axes_dim[i],  # 每个维度的旋转维度
                pos[..., i],
                theta=self.theta,  # 2000
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)

        freqs_cos = torch.cat(cos_out, dim=-1)
        freqs_sin = torch.cat(sin_out, dim=-1)

        return freqs_cos, freqs_sin
```

### 5.3 联合注意力机制

```python
class Flux2Attention(nn.Module):
    """
    图像-文本联合注意力

    流程:
    1. 图像tokens生成 Q_img, K_img, V_img
    2. 文本tokens生成 Q_txt, K_txt, V_txt
    3. 合并: Q = [Q_txt, Q_img], K = [K_txt, K_img], V = [V_txt, V_img]
    4. 计算注意力
    5. 分离输出
    """

    def forward(self, hidden_states, encoder_hidden_states):
        # 1. 图像QKV
        qkv, _ = self.to_qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        # 2. 文本QKV
        encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
        encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)

        # 3. RMS归一化
        query = self.norm_q(query.unflatten(-1, (heads, -1)))
        key = self.norm_k(key.unflatten(-1, (heads, -1)))
        encoder_query = self.norm_added_q(encoder_query.unflatten(-1, (heads, -1)))
        encoder_key = self.norm_added_k(encoder_key.unflatten(-1, (heads, -1)))

        # 4. 合并
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        # 5. 应用RoPE
        query, key = self.rope(query, key, image_rotary_emb)

        # 6. 计算注意力
        hidden_states = self.attn(query, key, value)

        # 7. 分离输出
        context_len = encoder_hidden_states.shape[1]
        encoder_hidden_states, hidden_states = hidden_states.split([context_len, ...])

        # 8. 输出投影
        hidden_states = self.to_out(hidden_states)
        encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
```

### 5.4 Classifier-Free Guidance (CFG)

```python
# CFG公式
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

# 在Flux2中的实现
if self.do_classifier_free_guidance:
    # 正向提示词
    positive_kwargs = {
        "hidden_states": latent_model_input,
        "encoder_hidden_states": prompt_embeds,  # 条件嵌入
        ...
    }

    # 空提示词
    negative_kwargs = {
        "hidden_states": latent_model_input,
        "encoder_hidden_states": negative_prompt_embeds,  # 无条件嵌入
        ...
    }

    noise_pred = self.predict_noise_maybe_with_cfg(
        do_true_cfg=True,
        true_cfg_scale=guidance_scale,
        positive_kwargs=positive_kwargs,
        negative_kwargs=negative_kwargs,
    )
```

---

## 6. vLLM-Omni优化技术

### 6.1 序列并行 (Sequence Parallelism)

```python
# SP配置
class DiffusionParallelConfig:
    sequence_parallel_size: int = None  # SP总大小
    ulysses_degree: int = 1             # Ulysses并行度
    ring_degree: int = 1                # Ring注意力并行度

    # 关系: sequence_parallel_size = ulysses_degree * ring_degree
```

**SP计划 (_sp_plan):**
```python
_sp_plan = {
    # 在根级别分片hidden_states
    "": {
        "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
    },
    # 在RoPE准备阶段分片图像频率
    "rope_prepare": {
        2: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),
        3: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),
    },
    # 在输出投影时收集结果
    "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
}
```

**Ulysses注意力流程:**
```
┌─────────────────────────────────────────────────────────────┐
│                    Ulysses SP Attention                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Rank 0: Q[0:N/sp], K[0:N/sp], V[0:N/sp]                   │
│  Rank 1: Q[N/sp:2N/sp], K[N/sp:2N/sp], V[N/sp:2N/sp]       │
│  ...                                                        │
│                                                             │
│  Step 1: AllToAll Q, K, V                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 每个rank获得所有序列位置的Q/K/V的子集              │   │
│  │ 用于计算局部注意力                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Step 2: 局部注意力计算                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Attn(Q, K, V) 在每个rank上独立计算                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Step 3: AllToAll Output                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 将输出重新组织回序列分片                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 CFG并行

```python
cfg_parallel_size: int = 1  # 1, 2, 或 3

# 当cfg_parallel_size=2时:
# - Rank 0: 计算条件噪声预测
# - Rank 1: 计算无条件噪声预测
# - AllReduce: 合并CFG结果

# 实现
class CFGParallelMixin:
    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict,
        negative_kwargs: dict,
    ):
        if do_true_cfg:
            # 并行计算正负条件
            if self.cfg_parallel_size > 1:
                # 在不同rank上并行计算
                positive_noise = self.transformer(**positive_kwargs)
                negative_noise = self.transformer(**negative_kwargs)
                # AllReduce合并
                noise_pred = negative_noise + true_cfg_scale * (positive_noise - negative_noise)
            else:
                # 顺序计算
                positive_noise = self.transformer(**positive_kwargs)
                negative_noise = self.transformer(**negative_kwargs)
                noise_pred = negative_noise + true_cfg_scale * (positive_noise - negative_noise)
        else:
            noise_pred = self.transformer(**positive_kwargs)

        return noise_pred
```

### 6.3 HSDP (Hybrid Sharded Data Parallel)

```python
# HSDP配置
use_hsdp: bool = False
hsdp_shard_size: int = -1      # 分片大小
hsdp_replicate_size: int = 1   # 副本大小

# HSDP允许:
# 1. 将模型权重分片到多个GPU
# 2. 保持多个完整副本用于数据并行
# 3. 减少每个GPU的内存占用

# 分片条件
_hsdp_shard_conditions = [
    lambda name, module: "transformer_blocks" in name or "single_transformer_blocks" in name
]
```

### 6.4 注意力后端优化

```python
class Attention(nn.Module):
    """
    支持多种注意力后端:
    - Flash Attention (FA)
    - SDPA (PyTorch原生)
    - Sage Attention
    - Ring Attention (分布式)
    """

    def __init__(self, num_heads, head_size, causal=False):
        # 选择最优后端
        self.attn_backend = get_attn_backend(-1)

        # 初始化实现
        self.attention = self.attn_backend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=1.0 / (head_size ** 0.5),
            causal=causal,
        )

        # SDPA回退 (用于float32)
        self.sdpa_fallback = SDPABackend.get_impl_cls()(...)

    def forward(self, query, key, value, attn_metadata=None):
        # 1. 预处理 (通信/重分片)
        query, key, value, attn_metadata, ctx = \
            self.parallel_strategy.pre_attention(query, key, value, attn_metadata)

        # 2. 注意力计算
        if self.use_ring:
            out = self._run_ring_attention(query, key, value, attn_metadata)
        else:
            out = self._run_local_attention(query, key, value, attn_metadata)

        # 3. 后处理 (反向通信)
        out = self.parallel_strategy.post_attention(out, ctx)

        return out
```

### 6.5 量化支持

```python
# 支持的量化方法
quantization_config: str | QuantizationConfig | dict | None

# 例如FP8量化
quantization_config = {
    "method": "fp8",
    "activation_scheme": "dynamic",
}

# 在Linear层中应用
class QKVParallelLinear(nn.Module):
    def __init__(self, hidden_size, head_size, num_heads, quant_config=None):
        if quant_config is not None:
            # 使用量化权重
            self.weight = quant_config.get_quantized_weight(...)
```

---

## 7. 矩阵形状示例

### 7.1 端到端形状追踪

假设生成一张 1024x1024 的图像，batch_size=1，50步推理：

```python
# ========== 输入阶段 ==========
# 文本输入
prompt = "A beautiful sunset over the ocean"
input_ids: [1, 512]  # 分词后

# 文本编码
# Qwen3ForCausalLM
hidden_states (layer 9):  [1, 512, 5120]
hidden_states (layer 18): [1, 512, 5120]
hidden_states (layer 27): [1, 512, 5120]
# 合并
prompt_embeds: [1, 512, 15360]  # 5120 * 3

# 位置ID
text_ids: [1, 512, 4]  # (T, H, W, L) 坐标

# ========== 潜在空间 ==========
# 噪声初始化
height = 1024 // 32 * 2 = 64
width = 1024 // 32 * 2 = 64
latents: [1, 512, 32, 32]  # patchify后: 128*4=512通道

# 打包
latents: [1, 1024, 512]  # 32*32=1024 tokens

# 位置ID
latent_ids: [1, 1024, 4]

# ========== Transformer ==========
# 输入嵌入
hidden_states: [1, 1024, 6144]  # inner_dim = 48*128
encoder_hidden_states: [1, 512, 6144]

# RoPE编码
freqs_cos: [1536, 128]  # (512+1024) * head_dim/2
freqs_sin: [1536, 128]

# Double-Stream Block
# Attention输入
query: [1, 1536, 48, 128]  # [B, seq, heads, head_dim]
key: [1, 1536, 48, 128]
value: [1, 1536, 48, 128]

# Attention输出
attn_output: [1, 1536, 6144]
# 分离
encoder_output: [1, 512, 6144]
hidden_output: [1, 1024, 6144]

# Single-Stream Block
# 合并输入
hidden_states: [1, 1536, 6144]
# QKV + MLP融合
qkv_mlp: [1, 1536, 6144*3 + 24576]  # 3*inner_dim + mlp_hidden
# 分离
qkv: [1, 1536, 18432]
mlp: [1, 1536, 24576]

# 输出
output: [1, 1024, 512]  # 移除文本tokens后

# ========== VAE解码 ==========
# 解包
latents: [1, 512, 32, 32]

# Unpatchify
latents: [1, 128, 64, 64]

# Batch Norm逆变换
latents = latents * bn_std + bn_mean

# VAE解码
image: [1, 3, 1024, 1024]

# ========== 输出 ==========
# 后处理
image = image.clip(0, 1)
image = (image * 255).to(uint8)
PIL.Image.fromarray(image)
```

### 7.2 关键操作的形状变化

```python
# ========== Patchify ==========
# 输入: [B, 32, 64, 64]  # VAE latent
# 输出: [B, 128, 32, 32]  # 4x通道，1/2空间

# 操作
x = x.view(B, 32, 32, 2, 32, 2)  # [B, C, H/2, 2, W/2, 2]
x = x.permute(0, 1, 3, 5, 2, 4)   # [B, C, 2, 2, H/2, W/2]
x = x.reshape(B, 128, 32, 32)     # [B, C*4, H/2, W/2]

# ========== Pack Latents ==========
# 输入: [B, 128, 32, 32]
# 输出: [B, 1024, 128]

x = x.reshape(B, 128, 1024)  # [B, C, H*W]
x = x.permute(0, 2, 1)        # [B, H*W, C]

# ========== QKV投影 ==========
# 输入: [B, seq_len, 6144]
# QKV输出: [B, seq_len, 18432]  # 6144 * 3

q, k, v = output.chunk(3, dim=-1)
# q, k, v: [B, seq_len, 6144] each

# 重塑为多头
q = q.view(B, seq_len, 48, 128)  # [B, seq, heads, head_dim]

# ========== 注意力计算 ==========
# Q, K, V: [B, seq, heads, head_dim]

# 缩放点积注意力
scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(head_dim)
# scores: [B, heads, seq, seq]

attn_weights = softmax(scores)
# attn_weights: [B, heads, seq, seq]

output = torch.matmul(attn_weights, V)
# output: [B, seq, heads, head_dim]

# 展平
output = output.flatten(2, 3)
# output: [B, seq, heads * head_dim]
```

### 7.3 内存分析

```python
# 假设配置:
# - 图像尺寸: 1024x1024
# - Batch size: 1
# - 精度: bfloat16 (2 bytes)

# ========== 主要内存占用 ==========

# 1. 模型权重
# Transformer: ~4B 参数 * 2 bytes = 8GB
# VAE: ~100M 参数 * 2 bytes = 200MB
# Text Encoder: ~7B 参数 * 2 bytes = 14GB (如果常驻GPU)

# 2. 激活值 (单步)
# - latents: [1, 1024, 512] * 2 = 1MB
# - prompt_embeds: [1, 512, 15360] * 2 = 15.7MB
# - 中间激活 (48 blocks): ~500MB

# 3. 注意力内存
# - 对于序列长度 1536 (512 text + 1024 image)
# - 注意力矩阵: [1, 48, 1536, 1536] * 2 = 226MB

# 总计峰值: ~10-15GB (不含Text Encoder)
```

---

## 附录

### A. 文件结构

```
vllm_omni/diffusion/models/flux2_klein/
├── __init__.py                    # 模块导出
├── flux2_klein_transformer.py     # Transformer模型定义
└── pipeline_flux2_klein.py        # 推理管道实现
```

### B. 依赖关系

```
Flux2KleinPipeline
├── Qwen3ForCausalLM (transformers)
├── Qwen2TokenizerFast (transformers)
├── AutoencoderKLFlux2 (diffusers)
├── FlowMatchEulerDiscreteScheduler (diffusers)
└── Flux2Transformer2DModel
    ├── Flux2PosEmbed
    ├── Flux2TimestepGuidanceEmbeddings
    ├── Flux2Modulation
    ├── Flux2TransformerBlock
    │   ├── Flux2Attention
    │   └── Flux2FeedForward
    └── Flux2SingleTransformerBlock
        └── Flux2ParallelSelfAttention
```

### C. 关键参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_layers | 8 | Double-stream块数量 |
| num_single_layers | 48 | Single-stream块数量 |
| attention_head_dim | 128 | 每个注意力头的维度 |
| num_attention_heads | 48 | 注意力头数量 |
| inner_dim | 6144 | 隐藏层维度 (48*128) |
| joint_attention_dim | 15360 | 文本编码器输出维度 |
| mlp_ratio | 3.0 | MLP扩展比例 |
| rope_theta | 2000 | RoPE基频 |
| axes_dims_rope | (32,32,32,32) | RoPE各维度 |

### D. 性能优化建议

1. **使用BF16精度**: 减少内存占用，加速计算
2. **启用Sequence Parallel**: 分布式处理长序列
3. **使用量化**: FP8/INT8量化减少显存
4. **启用编译**: `torch.compile` 加速Transformer块
5. **优化步数**: 根据质量需求调整推理步数 (通常20-50步)
