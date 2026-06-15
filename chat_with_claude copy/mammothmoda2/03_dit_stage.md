# DiT Stage (扩散阶段) 架构详解

## 1. 类层次结构

```
MammothModa2DiTPipeline
│
├── gen_vae: AutoencoderKL
│   ├── encoder
│   └── decoder
│
├── gen_transformer: Transformer2DModel
│   ├── x_embedder: Linear                    # 图像 token 嵌入
│   ├── ref_image_patch_embedder: Linear      # 参考图像嵌入
│   ├── time_caption_embed: CombinedEmbedding # 时间步 + 文本嵌入
│   ├── rope_embedder: RotaryPosEmbedReal     # 旋转位置编码
│   │
│   ├── noise_refiner: ModuleList[TransformerBlock × num_refiner_layers]
│   │   └── 处理噪声 latent tokens
│   │
│   ├── context_refiner: ModuleList[TransformerBlock × num_refiner_layers]
│   │   └── 处理文本条件 tokens
│   │
│   ├── layers: ModuleList[TransformerBlock × num_layers]
│   │   └── 主干 Transformer 层
│   │
│   ├── image_index_embedding: Parameter      # 多图像索引嵌入
│   └── norm_out: LuminaLayerNormContinuous   # 输出归一化
│
├── gen_image_condition_refiner: SimpleQFormerImageRefiner (可选)
│   └── 图像条件精炼器
│
└── gen_freqs_cis: List[Tuple[Tensor, Tensor]]
    └── 预计算的 RoPE 频率
```

## 2. 核心组件详解

### 2.1 MammothModa2DiTPipeline

**位置**: `pipeline_mammothmoda2_dit.py:22-309`

**职责**: DiT + VAE 生成管线

**关键属性**:
```python
self.config: Mammothmoda2Config       # 顶层配置
self.gen_vae: AutoencoderKL           # VAE 编解码器
self.gen_transformer: Transformer2DModel  # DiT 模型
self.gen_image_condition_refiner: SimpleQFormerImageRefiner  # 可选
self.gen_freqs_cis: list              # 预计算 RoPE
self._llm_hidden_size: int            # LLM 隐藏层维度
```

#### forward() - 主扩散循环

```python
def forward(
    self,
    inputs_embeds: torch.Tensor | None,
    **kwargs,
) -> OmniOutput:
    # 1. 从 runtime_additional_information 获取条件
    info = runtime_addi[0]
    text_cond = info["text_prompt_embeds"]      # [T_text, hidden_size]
    image_cond = info["image_prompt_embeds"]    # [T_img, hidden_size]
    image_hw = (height, width)
    text_guidance_scale = info["text_guidance_scale"]
    cfg_range = info["cfg_range"]
    num_inference_steps = info["num_inference_steps"]
    
    # 2. 可选: 图像条件精炼
    if self.gen_image_condition_refiner is not None:
        image_cond = self.gen_image_condition_refiner(image_cond)
    
    # 3. 构建条件嵌入
    prompt_embeds = torch.cat([text_cond, image_cond], dim=1)
    # prompt_embeds: [1, T_text + T_img, hidden_size]
    
    # 4. 初始化噪声 latent
    shape = (1, latent_channels, 2*H//16, 2*W//16)
    latents = randn_tensor(shape)
    # latents: [1, 16, 2H/16, 2W/16]
    
    # 5. 扩散循环
    for i, t in enumerate(scheduler.timesteps):
        # 5.1 DiT 前向
        model_pred = self.gen_transformer(
            hidden_states=latents,
            timestep=t,
            text_hidden_states=prompt_embeds,
            freqs_cis=self.gen_freqs_cis,
        )
        
        # 5.2 CFG (Classifier-Free Guidance)
        if guidance_scale > 1.0:
            model_pred_uncond = self.gen_transformer(
                hidden_states=latents,
                timestep=t,
                text_hidden_states=negative_prompt_embeds,
                freqs_cis=self.gen_freqs_cis,
            )
            model_pred = uncond + scale * (pred - uncond)
        
        # 5.3 Euler 步进
        latents = scheduler.step(model_pred, t, latents)
    
    # 6. VAE 解码
    image = self.gen_vae.decode(latents)
    # image: [1, 3, H, W]
    
    return OmniOutput(multimodal_outputs=image)
```

---

### 2.2 Transformer2DModel

**位置**: `mammothmoda2_dit_model.py:493-809`

**职责**: 主干扩散 Transformer

**关键属性**:
```python
self.hidden_size: int                    # 隐藏层维度
self.patch_size: int                     # Patch 大小 (默认 2)
self.rope_embedder: RotaryPosEmbedReal   # RoPE 嵌入器
self.x_embedder: Linear                  # 噪声图像嵌入
self.ref_image_patch_embedder: Linear    # 参考图像嵌入
self.time_caption_embed: CombinedEmbedding
self.noise_refiner: ModuleList           # 噪声精炼层
self.ref_image_refiner: ModuleList       # 参考图像精炼层
self.context_refiner: ModuleList         # 文本条件精炼层
self.layers: ModuleList                  # 主干 Transformer 层
self.norm_out: LuminaLayerNormContinuous # 输出归一化
```

#### forward() - 主前向传播

```python
def forward(
    self,
    hidden_states: torch.Tensor,         # [B, C, H, W] - 噪声 latent
    timestep: torch.Tensor,              # [B] - 时间步
    text_hidden_states: torch.Tensor,    # [B, T_text, hidden]
    text_attention_mask: torch.Tensor,   # [B, T_text]
    freqs_cis: torch.Tensor,             # 预计算 RoPE
    ref_image_hidden_states: None,       # 未使用
    return_dict: bool = False,
) -> torch.Tensor:
    
    batch_size, channels, height, width = hidden_states.shape
    p = self.config.patch_size
    
    # 1. 时间步 + 文本嵌入
    temb, text_hidden_states = self.time_caption_embed(
        timestep, text_hidden_states, hidden_states.dtype
    )
    # temb: [B, min(hidden_size, 1024)]
    # text_hidden_states: [B, T_text, hidden_size]
    
    # 2. 图像 Patch 嵌入
    img_tokens = rearrange(hidden_states, 
        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", 
        p1=p, p2=p)
    # img_tokens: [B, (H/p)*(W/p), p*p*C]
    
    img_tokens = self.x_embedder(img_tokens)
    # img_tokens: [B, num_patches, hidden_size]
    
    # 3. 计算 RoPE
    (context_rotary_emb, _, noise_rotary_emb, rotary_emb,
     encoder_seq_lengths, seq_lengths) = self.rope_embedder(...)
    
    # 4. 精炼器处理
    for layer in self.context_refiner:
        text_hidden_states = layer(
            text_hidden_states, text_attention_mask, context_rotary_emb
        )
    
    for layer in self.noise_refiner:
        img_tokens = layer(img_tokens, img_mask, noise_rotary_emb, temb)
    
    # 5. 拼接文本和图像 token
    joint_hidden_states = torch.cat([text_hidden_states, img_tokens], dim=1)
    # joint_hidden_states: [B, T_text + num_patches, hidden_size]
    
    # 6. 主干 Transformer 层
    for layer in self.layers:
        joint_hidden_states = layer(
            joint_hidden_states, attention_mask, rotary_emb, temb
        )
    
    # 7. 输出归一化和投影
    hidden_states = self.norm_out(joint_hidden_states, temb)
    # hidden_states: [B, seq_len, p*p*out_channels]
    
    # 8. 提取图像部分并 reshape
    img_hidden_states = hidden_states[:, encoder_seq_len:encoder_seq_len+img_len]
    output = rearrange(img_hidden_states,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=height//p, w=width//p, p1=p, p2=p)
    # output: [B, out_channels, H, W]
    
    return output
```

---

### 2.3 TransformerBlock

**位置**: `mammothmoda2_dit_model.py:407-490`

**职责**: 单个 Transformer 块

**关键属性**:
```python
self.head_dim: int                      # 每个头的维度
self.modulation: bool                   # 是否启用调制
self.attn: Attention                    # 自注意力层
self.feed_forward: LuminaFeedForward    # FFN 层
self.norm1: LuminaRMSNormZero | Qwen2RMSNorm  # 输入归一化
self.norm2: Qwen2RMSNorm                # 注意力后归一化
self.ffn_norm1: Qwen2RMSNorm            # FFN 前归一化
self.ffn_norm2: Qwen2RMSNorm            # FFN 后归一化
```

#### forward()

```python
def forward(
    self,
    hidden_states: torch.Tensor,      # [B, seq_len, dim]
    attention_mask: torch.Tensor,     # [B, seq_len]
    image_rotary_emb: torch.Tensor,   # RoPE 嵌入
    temb: torch.Tensor | None,        # 时间步嵌入 (调制模式)
) -> torch.Tensor:
    
    if self.modulation:
        # 调制模式: 使用 timestep 调制
        norm_hidden, gate_msa, scale_mlp, gate_mlp = self.norm1(
            hidden_states, temb
        )
        attn_out = self.attn(norm_hidden, ..., image_rotary_emb)
        hidden_states = hidden_states + gate_msa.tanh() * self.norm2(attn_out)
        
        mlp_out = self.feed_forward(
            self.ffn_norm1(hidden_states) * (1 + scale_mlp)
        )
        hidden_states = hidden_states + gate_mlp.tanh() * self.ffn_norm2(mlp_out)
    else:
        # 非调制模式
        norm_hidden = self.norm1(hidden_states)
        attn_out = self.attn(norm_hidden, ..., image_rotary_emb)
        hidden_states = hidden_states + self.norm2(attn_out)
        
        mlp_out = self.feed_forward(self.ffn_norm1(hidden_states))
        hidden_states = hidden_states + self.ffn_norm2(mlp_out)
    
    return hidden_states
```

---

### 2.4 关键嵌入组件

#### Lumina2CombinedTimestepCaptionEmbedding

**位置**: `mammothmoda2_dit_model.py:152-185`

```python
def forward(
    self,
    timestep: torch.Tensor,           # [B] - 时间步
    text_hidden_states: torch.Tensor, # [B, T, text_feat_dim]
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # 1. 时间步嵌入
    timestep_proj = self.time_proj(timestep)  # 正弦编码
    time_embed = self.timestep_embedder(timestep_proj)
    # time_embed: [B, min(hidden_size, 1024)]
    
    # 2. 文本嵌入
    caption_embed = self.caption_embedder(text_hidden_states)
    # caption_embed: [B, T, hidden_size]
    
    return time_embed, caption_embed
```

#### LuminaRMSNormZero

**位置**: `mammothmoda2_dit_model.py:32-64`

自适应 RMS 归一化，生成调制参数:

```python
def forward(
    self,
    x: torch.Tensor,      # [B, seq_len, dim]
    emb: torch.Tensor,    # [B, emb_dim] - 时间步嵌入
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    
    emb = self.linear(self.silu(emb))  # [B, 4*dim]
    scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
    # 每个: [B, dim]
    
    x = self.norm(x) * (1 + scale_msa[:, None])
    
    return x, gate_msa, scale_mlp, gate_mlp
```

---

### 2.5 SimpleQFormerImageRefiner

**位置**: `mammothmoda2_dit_model.py:188-274`

Q-Former 风格的图像条件精炼器

**结构**:
```
输入: [B, seq_len, hidden_size]
      │
      ▼ input_proj
      │ [B, seq_len, hidden_size]
      │
query: [B, num_queries, hidden_size] (可学习)
      │
      ▼ N × Decoder Layer
      │   - ln_q1 → self_attn → residual
      │   - ln_q2 → cross_attn(kv=输入) → residual  
      │   - ln_ffn → ffn → residual
      │
输出: [B, num_queries, hidden_size]
```

---

### 2.6 AttnProcessor

**位置**: `mammothmoda2_dit_model.py:277-404`

支持 Flash Attention 和 RoPE 的注意力处理器

```python
def __call__(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,          # [B, S, H*D]
    encoder_hidden_states: torch.Tensor,  # [B, S, H*D]
    attention_mask: torch.Tensor,         # [B, S]
    image_rotary_emb: torch.Tensor,       # RoPE
) -> torch.Tensor:
    
    # 1. QKV 投影
    query = attn.to_q(hidden_states)    # [B, S, H*D]
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    
    # 2. Reshape 为多头形式
    query = query.view(B, S, heads, head_dim)
    key = key.view(B, S, kv_heads, head_dim)
    value = value.view(B, S, kv_heads, head_dim)
    
    # 3. QK 归一化
    query = attn.norm_q(query)
    key = attn.norm_k(key)
    
    # 4. 应用 RoPE
    query = apply_real_rotary_emb(query, freqs_cos, freqs_sin)
    key = apply_real_rotary_emb(key, freqs_cos, freqs_sin)
    
    # 5. 注意力计算 (Flash Attention 或 SDPA)
    output = flash_attn_varlen_func(...) 或 F.scaled_dot_product_attention(...)
    
    # 6. 输出投影
    output = attn.to_out[0](output)
    output = attn.to_out[1](output)
    
    return output
```

---

## 3. 张量形状追踪

### 3.1 完整扩散流程

```
输入条件:
  text_prompt_embeds:           [T_text, llm_hidden_size]
  image_prompt_embeds:          [T_img, llm_hidden_size]
  image_size:                   (H, W)

                    │ 条件准备
text_embeds:                    [1, T_text, hidden_size]
image_embeds:                   [1, T_img, hidden_size]
prompt_embeds:                  [1, T_text + T_img, hidden_size]

                    │ 初始化噪声
latents:                        [1, 16, 2H/16, 2W/16]
                                = [1, 16, H/8, W/8]

                    │ 扩散循环 (每个时间步)
                    │
                    ▼ Patch 嵌入
img_patches:                    [1, (H/8)*(W/8), patch_size²*16]
                                = [1, (H/8)*(W/8), 64]  (patch_size=2)
                    │
                    ▼ x_embedder
img_tokens:                     [1, num_patches, hidden_size]
                                = [1, (H/16)*(W/16), hidden_size]

                    ▼ 时间步 + 文本嵌入
temb:                           [1, min(hidden_size, 1024)]
text_hidden_states:             [1, T_text, hidden_size]

                    ▼ Refiners
context_refined:                [1, T_text, hidden_size]
noise_refined:                  [1, num_patches, hidden_size]

                    ▼ 拼接
joint_hidden:                   [1, T_text + num_patches, hidden_size]

                    ▼ N × TransformerBlock
transformer_out:                [1, T_text + num_patches, hidden_size]

                    ▼ norm_out + 提取图像部分
img_output:                     [1, num_patches, patch_size²*out_channels]

                    ▼ reshape
output:                         [1, out_channels, H/8, W/8]

                    │ Euler 步进
latents:                        [1, out_channels, H/8, W/8]

                    │ 循环结束，VAE 解码
                    ▼
latents_scaled:                 [1, 16, H/8, W/8] (如有 scaling_factor)
image:                          [1, 3, H, W]
```

### 3.2 RoPE 位置编码维度

```python
# 配置
axes_dim_rope = (32, 32, 32)    # 每个轴的维度
axes_lens = (10000, 10000, 10000)  # 每个轴的最大长度

# 总维度 = sum(axes_dim_rope) = 96
# 必须满足: hidden_size // num_heads == sum(axes_dim_rope)

# 位置 ID 格式: [batch, seq_len, 3]
# position_ids[:, :, 0] = 时间位置
# position_ids[:, :, 1] = 行位置
# position_ids[:, :, 2] = 列位置
```

---

## 4. 配置参数

### 4.1 Transformer2DModel 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `patch_size` | 2 | Patch 大小 |
| `in_channels` | 16 | 输入通道数 (latent) |
| `out_channels` | 16 | 输出通道数 |
| `hidden_size` | 2304 | 隐藏层维度 |
| `num_layers` | 26 | 主干层数 |
| `num_refiner_layers` | 2 | 精炼器层数 |
| `num_attention_heads` | 24 | 注意力头数 |
| `num_kv_heads` | 8 | KV 头数 (GQA) |
| `axes_dim_rope` | (32, 32, 32) | RoPE 各轴维度 |
| `text_feat_dim` | 1024 | 文本特征维度 |

### 4.2 VAE 配置 (AutoencoderKL)

| 参数 | 说明 |
|------|------|
| `scaling_factor` | Latent 缩放因子 |
| `shift_factor` | Latent 偏移因子 |
| `latent_channels` | Latent 通道数 (16) |

---

## 5. FlowMatchEulerDiscreteScheduler

**位置**: `schedulers.py:40-137`

Rectified Flow / Flow Matching 调度器

```python
class FlowMatchEulerDiscreteScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        dynamic_time_shift: bool = True,
    ):
        # 时间步范围: [0, 1]
        self.timesteps = torch.linspace(0, 1, num_train_timesteps)
    
    def set_timesteps(
        self,
        num_inference_steps: int,
        num_tokens: int | None = None,  # 用于动态时间偏移
    ):
        if self.dynamic_time_shift and num_tokens is not None:
            # 根据 token 数量调整时间步
            m = sqrt(num_tokens) / 40.0
            timesteps = timesteps / (m - m * timesteps + timesteps)
    
    def step(
        self,
        model_output: torch.Tensor,  # 模型预测的速度场
        timestep: float,
        sample: torch.Tensor,        # 当前样本
    ) -> torch.Tensor:
        t = self.timesteps[step_index]
        t_next = self.timesteps[step_index + 1]
        
        # Euler 步进: x_{t+1} = x_t + (t_{t+1} - t_t) * v_t
        prev_sample = sample + (t_next - t) * model_output
        
        return prev_sample
```

---

## 6. 权重加载

DiT 阶段权重前缀映射:

| Checkpoint 前缀 | vLLM 前缀 |
|----------------|-----------|
| `gen_transformer.` | `gen_transformer.` |
| `gen_vae.` | `gen_vae.` |
| `gen_image_condition_refiner.` | `gen_image_condition_refiner.` |
| `llm_model.` | 跳过 (AR 阶段) |
