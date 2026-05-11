# OvisImage 文生图模型全结构分析

## 1. 模型概述

OvisImage 是由阿里巴巴推出的文生图扩散模型，基于 DiT (Diffusion Transformer) 架构，采用 Flow Matching 训练方式。该模型借鉴了 Flux 的架构设计，支持高质量文本到图像的生成。

### 1.1 核心组件架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OvisImagePipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────────────────┐  │
│  │ Text Encoder│  │ Transformer      │  │ VAE Decoder                   │  │
│  │ (Qwen3)     │→ │ (OvisImage       │→ │ (AutoencoderKL)               │  │
│  │             │  │  Transformer2D)  │  │                               │  │
│  └─────────────┘  └──────────────────┘  └───────────────────────────────┘  │
│         ↓                  ↓                          ↓                     │
│  [Text Embeddings]  [Denoised Latents]       [Generated Image]             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline 完整推理流程

### 2.1 推理流程概览

```
输入文本 Prompt
       │
       ▼
┌──────────────────┐
│ 1. 文本编码      │  Qwen3 Text Encoder + Tokenizer
│    encode_prompt │  → prompt_embeds [B, 256, 2048]
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ 2. 准备 Latents  │  初始化高斯噪声
│ prepare_latents  │  → latents [B, H/16*W/16, 64]
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ 3. 准备 Timesteps│  FlowMatchEulerDiscreteScheduler
│prepare_timesteps │  → timesteps [num_steps]
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ 4. 扩散去噪循环  │  Transformer 前向传播
│    diffuse       │  (num_inference_steps 次)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ 5. 解码图像      │  VAE Decoder
│    VAE.decode    │  → image [B, 3, H, W]
└──────────────────┘
       │
       ▼
输出图像 PIL.Image
```

### 2.2 详细推理步骤

#### Step 1: 文本编码 (`encode_prompt`)

**输入**:
- `prompt`: str 或 List[str] - 用户输入的文本提示

**处理流程**:
1. 构建 chat message 格式，添加系统提示词
2. 使用 Qwen2TokenizerFast 进行分词
3. Qwen3 Text Encoder 编码得到 hidden states
4. 应用 attention mask 并截取有效 token

**代码路径**: `pipeline_ovis_image.py:262-297`

**输出**:
- `prompt_embeds`: [B, seq_len, 2048] - 文本嵌入向量
- `text_ids`: [seq_len, 3] - 文本位置 ID

---

#### Step 2: 准备 Latents (`prepare_latents`)

**输入**:
- `batch_size`: 批次大小
- `height`, `width`: 目标图像尺寸
- `num_channel_latents`: 16 (= 64/4)

**处理流程**:
1. 计算 latent 空间尺寸 (缩小 16 倍)
2. 生成高斯噪声 `randn_tensor`
3. Pack latents 为 patch 序列格式

**代码路径**: `pipeline_ovis_image.py:381-413`

**输出**:
- `latents`: [B, H_latent/2 * W_latent/2, 64] - packed latent tensor
- `latent_image_ids`: [H_latent/2 * W_latent/2, 3] - 图像位置 ID

---

#### Step 3: 准备 Timesteps (`prepare_timesteps`)

**输入**:
- `num_inference_steps`: 去噪步数 (默认 50)
- `image_seq_len`: 图像序列长度

**处理流程**:
1. 计算 shift 参数 (动态时间步调整)
2. 使用 FlowMatchEulerDiscreteScheduler 设置时间步

**代码路径**: `pipeline_ovis_image.py:415-435`

**输出**:
- `timesteps`: [num_inference_steps] - 时间步序列

---

#### Step 4: 扩散去噪循环 (`diffuse`)

**输入**:
- `latents`: 噪声 latents
- `timesteps`: 时间步序列
- `prompt_embeds`: 文本嵌入

**处理流程**:
```python
for t in timesteps:
    # 1. 构建输入字典
    positive_kwargs = {
        "hidden_states": latents,
        "timestep": t / 1000,
        "encoder_hidden_states": prompt_embeds,
        "txt_ids": text_ids,
        "img_ids": latent_image_ids,
    }

    # 2. Transformer 预测噪声
    noise_pred = self.transformer(**positive_kwargs)

    # 3. CFG (Classifier-Free Guidance)
    if do_true_cfg:
        noise_pred = guidance_scale * positive_pred + (1 - guidance_scale) * negative_pred

    # 4. Scheduler 更新 latents
    latents = self.scheduler.step(noise_pred, t, latents)
```

**代码路径**: `pipeline_ovis_image.py:437-509`

---

#### Step 5: VAE 解码

**输入**:
- `latents`: 去噪后的 latents [B, seq_len, 64]

**处理流程**:
1. Unpack latents 为标准形状
2. 反归一化 (scaling_factor, shift_factor)
3. VAE decoder 解码为图像

**代码路径**: `pipeline_ovis_image.py:738-740`

**输出**:
- `image`: [B, 3, H, W] - 生成的图像张量

---

## 3. Transformer 架构详解

### 3.1 OvisImageTransformer2DModel 完整结构

```
OvisImageTransformer2DModel
│
├── pos_embed: OvisImagePosEmbed          # 旋转位置编码
│
├── time_proj: Timesteps                  # 时间步投影
├── timestep_embedder: TimestepEmbedding  # 时间嵌入 [256 → 3072]
│
├── context_embedder_norm: RMSNorm        # 文本嵌入归一化
├── context_embedder: Linear              # 文本嵌入投影 [2048 → 3072]
├── x_embedder: Linear                    # 图像嵌入投影 [64 → 3072]
│
├── transformer_blocks: ModuleList[OvisImageTransformerBlock]  # 双流 DiT 块 × 6
│
├── single_transformer_blocks: ModuleList[OvisImageSingleTransformerBlock]  # 单流 DiT 块 × 27
│
├── norm_out: AdaLayerNormContinuous      # 输出归一化
└── proj_out: Linear                      # 输出投影 [3072 → 64]
```

**代码路径**: `ovis_image_transformer.py:340-505`

### 3.2 Transformer 前向传播流程

```python
def forward(hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
    # 1. 图像嵌入
    hidden_states = self.x_embedder(hidden_states)  # [B, seq_len, 3072]

    # 2. 时间步嵌入
    timesteps_proj = self.time_proj(timestep)  # [B, 256]
    temb = self.timestep_embedder(timesteps_proj)  # [B, 3072]

    # 3. 文本嵌入
    encoder_hidden_states = self.context_embedder_norm(encoder_hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)  # [B, txt_seq, 3072]

    # 4. 计算 RoPE
    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)  # (cos, sin)

    # 5. 双流 DiT 块 (6层)
    for block in self.transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states, encoder_hidden_states, temb, image_rotary_emb
        )

    # 6. 单流 DiT 块 (27层)
    for block in self.single_transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states, encoder_hidden_states, temb, image_rotary_emb
        )

    # 7. 输出
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)  # [B, seq_len, 64]

    return output
```

---

## 4. 核心组件详解

### 4.1 OvisImageTransformerBlock (双流 DiT 块)

**作用**: 同时处理文本和图像信息，实现跨模态交互

**结构**:
```
OvisImageTransformerBlock
│
├── norm1: AdaLayerNormZero          # 图像分支自适应层归一化
├── norm1_context: AdaLayerNormZero  # 文本分支自适应层归一化
│
├── attn: OvisImageAttention         # 联合注意力 (文本+图像)
│
├── norm2: LayerNorm                 # 图像 FFN 前归一化
├── ff: FeedForward (SwiGLU)         # 图像前馈网络
│
├── norm2_context: LayerNorm         # 文本 FFN 前归一化
└── ff_context: FeedForward (SwiGLU) # 文本前馈网络
```

**前向传播**:
```python
def forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb):
    # 1. 自适应层归一化
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
        self.norm1_context(encoder_hidden_states, temb)

    # 2. 联合注意力
    attn_output, context_attn_output = self.attn(
        norm_hidden_states, norm_encoder_hidden_states, image_rotary_emb
    )

    # 3. 图像分支更新
    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hidden_states)

    # 4. 文本分支更新
    encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_encoder_hidden_states)

    return encoder_hidden_states, hidden_states
```

**输入输出形状**:
| 张量 | 输入形状 | 输出形状 |
|------|----------|----------|
| hidden_states | [B, img_seq, 3072] | [B, img_seq, 3072] |
| encoder_hidden_states | [B, txt_seq, 3072] | [B, txt_seq, 3072] |
| temb | [B, 3072] | - |

**代码路径**: `ovis_image_transformer.py:224-308`

---

### 4.2 OvisImageSingleTransformerBlock (单流 DiT 块)

**作用**: 在统一序列上处理文本和图像，更高效的推理

**结构**:
```
OvisImageSingleTransformerBlock
│
├── norm: AdaLayerNormZeroSingle  # 统一自适应层归一化
├── proj_mlp: Linear              # MLP 投影 [dim → dim*4*2]
├── act_mlp: SiLU                 # 激活函数
├── proj_out: Linear              # 输出投影
│
└── attn: OvisImageAttention      # 自注意力 (pre_only=True)
```

**前向传播**:
```python
def forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb):
    text_seq_len = encoder_hidden_states.shape[1]

    # 1. 拼接文本和图像
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # 2. 自适应归一化 + 门控
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, temb)

    # 3. MLP 分支 (SwiGLU)
    mlp_hidden_states, mlp_hidden_gate = self.proj_mlp(norm_hidden_states).chunk(2, dim=-1)
    mlp_hidden_states = self.act_mlp(mlp_hidden_gate) * mlp_hidden_states

    # 4. 注意力
    attn_output = self.attn(norm_hidden_states, image_rotary_emb)

    # 5. 合并 + 残差连接
    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    hidden_states = residual + gate.unsqueeze(1) * self.proj_out(hidden_states)

    # 6. 分离文本和图像
    encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]

    return encoder_hidden_states, hidden_states
```

**代码路径**: `ovis_image_transformer.py:169-221`

---

### 4.3 OvisImageAttention (注意力模块)

**作用**: 实现带 RoPE 的联合注意力机制

**结构**:
```
OvisImageAttention
│
├── norm_q, norm_k: RMSNorm           # Q/K 归一化
├── to_qkv: QKVParallelLinear         # QKV 投影 [dim → 3*dim]
│
├── norm_added_q, norm_added_k: RMSNorm  # 编码器 Q/K 归一化
├── add_kv_proj: QKVParallelLinear    # 编码器 QKV 投影
├── to_add_out: ReplicatedLinear      # 编码器输出投影
│
├── rope: RotaryEmbedding             # 旋转位置编码
├── attn: Attention                   # 注意力实现
│
└── to_out: [Linear, Dropout]         # 输出投影
```

**前向传播**:
```python
def forward(hidden_states, encoder_hidden_states, image_rotary_emb):
    # 1. 图像 QKV 投影
    qkv = self.to_qkv(hidden_states)
    query, key, value = qkv.chunk(3, dim=-1)

    # 2. Reshape 为多头格式
    query = query.unflatten(-1, (self.heads, -1))  # [B, seq, heads, head_dim]
    key = key.unflatten(-1, (self.heads, -1))
    value = value.unflatten(-1, (self.heads, -1))

    # 3. RMSNorm 归一化
    query = self.norm_q(query)
    key = self.norm_k(key)

    # 4. 文本 QKV 投影 (如果有)
    if encoder_hidden_states is not None:
        encoder_qkv = self.add_kv_proj(encoder_hidden_states)
        encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)

        encoder_query = self.norm_added_q(encoder_query.unflatten(-1, (self.heads, -1)))
        encoder_key = self.norm_added_k(encoder_key.unflatten(-1, (self.heads, -1)))

        # 拼接文本和图像
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

    # 5. 应用 RoPE
    if image_rotary_emb is not None:
        cos, sin = image_rotary_emb
        query = self.rope(query, cos, sin)
        key = self.rope(key, cos, sin)

    # 6. 注意力计算
    hidden_states = self.attn(query, key, value)
    hidden_states = hidden_states.flatten(2, 3)

    # 7. 分离并投影输出
    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(...)
        hidden_states = self.to_out[0](hidden_states)
        encoder_hidden_states = self.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states

    return hidden_states
```

**注意力计算细节**:
- 使用 scaled dot-product attention: `softmax(QK^T / sqrt(d)) * V`
- softmax_scale = 1.0 / sqrt(head_dim)
- causal = False (双向注意力)

**输入输出形状**:
| 张量 | 形状 |
|------|------|
| hidden_states (输入) | [B, img_seq, 3072] |
| encoder_hidden_states | [B, txt_seq, 3072] |
| query | [B, txt_seq + img_seq, 24, 128] |
| key | [B, txt_seq + img_seq, 24, 128] |
| value | [B, txt_seq + img_seq, 24, 128] |
| output | [B, img_seq, 3072] |

**代码路径**: `ovis_image_transformer.py:40-166`

---

### 4.4 OvisImagePosEmbed (位置编码)

**作用**: 生成 3D 旋转位置编码 (时间, 高度, 宽度)

**结构**:
```python
class OvisImagePosEmbed:
    def __init__(self, theta=10000, axes_dim=[16, 56, 56]):
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        # ids: [seq_len, 3] - (t, h, w) 位置
        # 输出: (cos, sin) 各 [seq_len, 128]
```

**位置 ID 格式**:
- 文本: `[0, token_idx, token_idx]` - 第 0 维全为 0
- 图像: `[0, height_idx, width_idx]` - 空间位置编码

**RoPE 维度分配**:
- 第 0 轴 (时间): 16 维
- 第 1 轴 (高度): 56 维
- 第 2 轴 (宽度): 56 维
- 总计: 128 维 (= head_dim)

**代码路径**: `ovis_image_transformer.py:311-337`

---

## 5. 数据流与张量形状变换

### 5.1 完整数据流

```
Prompt Text
    │
    ▼
Tokenizer: "a beautiful cat"
    │
    ▼ tokens: [B, seq_len]
    │
    ▼
Qwen3 Text Encoder
    │
    ▼ prompt_embeds: [B, 256, 2048]
    │
    ▼
context_embedder (Linear 2048→3072)
    │
    ▼ encoder_hidden_states: [B, 256, 3072]

Noise latents: [B, 64, H/16, W/16]
    │
    ▼ pack_latents
    │
    ▼ hidden_states: [B, (H/16*W/16)/4, 64]
    │
    ▼
x_embedder (Linear 64→3072)
    │
    ▼ hidden_states: [B, img_seq, 3072]

Timestep t: scalar
    │
    ▼
time_proj: [B, 256]
    │
    ▼
timestep_embedder: [B, 3072]
    │
    ▼ temb: [B, 3072]

┌─────────────────────────────────────────┐
│         Transformer Blocks              │
│                                         │
│  6 × OvisImageTransformerBlock         │
│  (双流: 文本和图像分别处理，交叉注意力)   │
│                                         │
│  27 × OvisImageSingleTransformerBlock  │
│  (单流: 文本和图像拼接后统一处理)        │
└─────────────────────────────────────────┘
    │
    ▼ hidden_states: [B, img_seq, 3072]
    │
    ▼
norm_out (AdaLayerNormContinuous)
    │
    ▼
proj_out (Linear 3072→64)
    │
    ▼ output: [B, img_seq, 64]
    │
    ▼
unpack_latents
    │
    ▼ latents: [B, 16, H/16, W/16]
    │
    ▼
VAE Decoder
    │
    ▼ image: [B, 3, H, W]
```

### 5.2 关键张量形状总结

| 组件 | 输入形状 | 输出形状 |
|------|----------|----------|
| Tokenizer | str | [seq_len] |
| Text Encoder | [B, seq_len] | [B, seq_len, 2048] |
| context_embedder | [B, 256, 2048] | [B, 256, 3072] |
| Latent Init | - | [B, 64, H/16, W/16] |
| pack_latents | [B, 64, H/16, W/16] | [B, H*W/256, 64] |
| x_embedder | [B, img_seq, 64] | [B, img_seq, 3072] |
| time_proj | [B] | [B, 256] |
| timestep_embedder | [B, 256] | [B, 3072] |
| TransformerBlock | [B, seq, 3072] | [B, seq, 3072] |
| proj_out | [B, img_seq, 3072] | [B, img_seq, 64] |
| unpack_latents | [B, img_seq, 64] | [B, 16, H/16, W/16] |
| VAE Decoder | [B, 16, H/16, W/16] | [B, 3, H, W] |

---

## 6. 模型超参数

### 6.1 默认配置

```python
# Transformer 配置
patch_size = 1
in_channels = 64
out_channels = 64
num_layers = 6           # 双流 DiT 块数量
num_single_layers = 27   # 单流 DiT 块数量
attention_head_dim = 128
num_attention_heads = 24
joint_attention_dim = 2048

# 计算出的维度
inner_dim = num_attention_heads * attention_head_dim  # 24 * 128 = 3072

# RoPE 配置
axes_dims_rope = (16, 56, 56)  # 时间、高度、宽度维度
theta = 10000

# VAE 配置
vae_scale_factor = 8  # VAE 下采样倍率

# 文本编码器
tokenizer_max_length = 283  # 256 + 27 (系统提示词 + user prompt begin id)
```

### 6.2 生成配置

```python
# 默认生成参数
height = 1024
width = 1024
num_inference_steps = 50
guidance_scale = 5.0
num_images_per_prompt = 1

# Scheduler 配置
base_image_seq_len = 256
max_image_seq_len = 4096
base_shift = 0.5
max_shift = 1.15
```

---

## 7. 特殊机制详解

### 7.1 Latent Packing 机制

**作用**: 将 2D latent 特征转换为 1D 序列，同时保留局部结构

**实现**:
```python
def _pack_latents(latents, batch_size, num_channel_latents, height, width):
    # 输入: [B, C, H, W]
    # 输出: [B, (H/2)*(W/2), C*4]

    latents = latents.view(batch_size, num_channel_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)  # [B, H/2, W/2, C, 2, 2]
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channel_latents * 4)
    return latents
```

**可视化**:
```
原始 latent [B, 64, H, W]:
┌──┬──┐
│A │B │  每个 2x2 patch 被打包成一个 token
├──┼──┤  64 channels → 64*4 = 256 features per token
│C │D │
└──┴──┘

Packed latent [B, H*W/4, 256]:
[A packed] [B packed] [C packed] [D packed] ...
```

### 7.2 Classifier-Free Guidance (CFG)

**作用**: 增强生成结果与文本提示的一致性

**实现**:
```python
# 正向条件
noise_pred_positive = transformer(latents, timestep, positive_prompt_embeds)

# 无条件 (空提示词)
noise_pred_negative = transformer(latents, timestep, negative_prompt_embeds)

# CFG 组合
noise_pred = noise_pred_negative + guidance_scale * (noise_pred_positive - noise_pred_negative)
```

**参数**:
- `guidance_scale = 5.0`: 默认值，越大越遵循文本提示

### 7.3 动态时间步 Shift

**作用**: 根据图像尺寸动态调整时间步分布

**实现**:
```python
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu
```

**原理**: 大图像需要更多去噪步数，shift 参数调整时间步密度分布

---

## 8. 权重加载机制

### 8.1 Stacked Parameters Mapping

```python
stacked_params_mapping = [
    # Self Attention
    (".to_qkv", ".to_q", "q"),
    (".to_qkv", ".to_k", "k"),
    (".to_qkv", ".to_v", "v"),
    # Cross Attention
    (".add_kv_proj", ".add_q_proj", "q"),
    (".add_kv_proj", ".add_k_proj", "k"),
    (".add_kv_proj", ".add_v_proj", "v"),
]
```

**说明**: Q、K、V 权重在加载时自动堆叠为单个 `to_qkv` 矩阵，提高推理效率

### 8.2 加载流程

```python
def load_weights(self, weights):
    for name, loaded_weight in weights:
        # 检查是否为 stacked parameter
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name in name:
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
        else:
            # 普通参数
            param = params_dict[name]
            param.weight_loader(param, loaded_weight)
```

---

## 9. 并行支持

### 9.1 Attention 并行策略

通过 `vllm_omni.diffusion.attention.layer.Attention` 支持:

1. **Ulysses Sequence Parallel**: 序列维度切分
2. **Ring Attention**: 环形注意力，支持超长序列
3. **Tensor Parallel**: 通过 `QKVParallelLinear` 实现

### 9.2 CFG Parallel

通过 `CFGParallelMixin` 实现 CFG 并行计算:

```python
class OvisImagePipeline(nn.Module, CFGParallelMixin, ...):
    def predict_noise_maybe_with_cfg(self, do_true_cfg, guidance_scale,
                                      positive_kwargs, negative_kwargs):
        if do_true_cfg:
            # 并行计算正向和无条件预测
            ...
```

---

## 10. 与 Flux 架构对比

| 特性 | OvisImage | Flux |
|------|-----------|------|
| 文本编码器 | Qwen3 (2B) | T5 XXL (4.7B) + CLIP |
| Transformer 层数 | 6 + 27 | 19 + 38 |
| 隐藏维度 | 3072 | 3072 |
| 注意力头数 | 24 | 24 |
| 头维度 | 128 | 128 |
| RoPE 维度 | (16, 56, 56) | (16, 56, 56) |
| VAE 压缩 | 8x (FLUX.1) | 8x (FLUX.1) |
| 训练方式 | Flow Matching | Flow Matching |

---

## 11. 总结

OvisImage 是一个基于 DiT 架构的文生图模型，核心特点:

1. **双阶段 Transformer**: 6 层双流块 + 27 层单流块
2. **Qwen3 文本编码器**: 中文理解能力强
3. **3D RoPE**: 时间、高度、宽度三维位置编码
4. **Latent Packing**: 高效的 2D→1D 转换
5. **Flow Matching**: 使用 FlowMatchEulerDiscreteScheduler
6. **CFG 支持**: 增强文本控制能力

模型通过 vllm-omni 框架实现了高效的推理服务，支持 Tensor Parallel、Sequence Parallel 等多种并行策略。
