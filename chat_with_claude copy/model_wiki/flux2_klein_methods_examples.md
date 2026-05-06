# Flux2-Klein 关键方法输入输出详解

## 目录

1. [Pipeline层方法](#1-pipeline层方法)
2. [Transformer层方法](#2-transformer层方法)
3. [注意力机制方法](#3-注意力机制方法)
4. [位置编码方法](#4-位置编码方法)
5. [完整推理示例](#5-完整推理示例)

---

## 1. Pipeline层方法

### 1.1 `Flux2KleinPipeline.forward()`

**功能**: 端到端图像生成的主入口

**输入参数**:
```python
def forward(
    self,
    req: OmniDiffusionRequest,     # 请求对象，包含所有生成参数
    image: PIL.Image | None,        # 可选的输入图像（用于图生图）
    prompt: str | list[str],        # 文本提示词
    height: int | None = None,      # 输出高度 (默认1024)
    width: int | None = None,       # 输出宽度 (默认1024)
    num_inference_steps: int = 50,  # 去噪步数
    guidance_scale: float = 4.0,    # CFG引导强度
    num_images_per_prompt: int = 1, # 每个提示词生成图像数
    generator: torch.Generator,      # 随机数生成器
    max_sequence_length: int = 512, # 文本最大长度
) -> DiffusionOutput:
```

**输入形状示例**:
```python
# 示例输入
req = OmniDiffusionRequest(
    prompts=["A cat sitting on a chair"],
    sampling_params=SamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=4.0,
    )
)

# 解析后的参数
prompt = "A cat sitting on a chair"
height = 1024
width = 1024
num_inference_steps = 50
guidance_scale = 4.0
```

**输出**:
```python
@dataclass
class DiffusionOutput:
    output: torch.Tensor | PIL.Image  # 生成的图像
    stage_durations: dict[str, float] # 各阶段耗时
```

**输出形状**:
```python
# PIL.Image模式
output = PIL.Image.Image  # RGB图像，尺寸1024x1024

# Tensor模式
output = torch.Tensor     # [B, 3, 1024, 1024], float32, [0, 1]
```

---

### 1.2 `encode_prompt()`

**功能**: 将文本提示词编码为嵌入向量

**输入**:
```python
def encode_prompt(
    self,
    prompt: str | list[str],        # 文本提示词
    device: torch.device,           # 目标设备
    num_images_per_prompt: int = 1, # 每个提示词生成图像数
    max_sequence_length: int = 512, # 最大序列长度
    text_encoder_out_layers: tuple = (9, 18, 27),  # 输出层
) -> tuple[torch.Tensor, torch.Tensor]:
```

**输入形状示例**:
```python
prompt = "A beautiful sunset over mountains"
device = torch.device("cuda:0")
max_sequence_length = 512
```

**输出**:
```python
# prompt_embeds: 文本嵌入
# text_ids: 文本位置ID

prompt_embeds: torch.Tensor  # [B, 512, 15360]
text_ids: torch.Tensor       # [B, 512, 4]
```

**详细计算过程**:
```python
# Step 1: 应用聊天模板
messages = [{"role": "user", "content": "A beautiful sunset over mountains"}]
text = tokenizer.apply_chat_template(messages, tokenize=False)
# text: "<|im_start|>user\nA beautiful sunset over mountains<|im_end|>\n<|im_start|>assistant\n"

# Step 2: 分词
inputs = tokenizer(text, max_length=512, padding="max_length", return_tensors="pt")
input_ids: torch.Tensor      # [1, 512]
attention_mask: torch.Tensor # [1, 512]

# Step 3: 文本编码器前向传播
output = text_encoder(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,
)

# Step 4: 提取多层隐藏状态
hidden_states_9  = output.hidden_states[9]   # [1, 512, 5120]
hidden_states_18 = output.hidden_states[18]  # [1, 512, 5120]
hidden_states_27 = output.hidden_states[27]  # [1, 512, 5120]

# Step 5: 堆叠并重塑
out = torch.stack([hidden_states_9, hidden_states_18, hidden_states_27], dim=1)
# out: [1, 3, 512, 5120]

out = out.permute(0, 2, 1, 3)
# out: [1, 512, 3, 5120]

prompt_embeds = out.reshape(1, 512, 15360)
# prompt_embeds: [1, 512, 15360]
```

---

### 1.3 `prepare_latents()`

**功能**: 初始化潜在空间表示

**输入**:
```python
def prepare_latents(
    self,
    batch_size: int,              # 批大小
    num_latents_channels: int,    # 潜在通道数 (128/4=32)
    height: int,                  # 图像高度
    width: int,                   # 图像宽度
    dtype: torch.dtype,           # 数据类型
    device: torch.device,         # 目标设备
    generator: torch.Generator,   # 随机数生成器
) -> tuple[torch.Tensor, torch.Tensor]:
```

**输入形状示例**:
```python
batch_size = 1
num_latents_channels = 32  # in_channels // 4
height = 1024
width = 1024
dtype = torch.bfloat16
device = torch.device("cuda:0")
```

**输出**:
```python
latents: torch.Tensor    # [B, H*W/64, C*4]
latent_ids: torch.Tensor # [B, H*W/64, 4]
```

**详细计算过程**:
```python
# Step 1: 计算潜在空间尺寸
vae_scale_factor = 16  # VAE下采样倍数
height = 2 * (1024 // 32)  # = 64
width = 2 * (1024 // 32)   # = 64

# Step 2: 生成噪声
shape = (1, 32 * 4, 64 // 2, 64 // 2)  # (1, 128, 32, 32)
latents = randn_tensor(shape, generator=generator)
# latents: [1, 128, 32, 32]

# Step 3: 准备位置ID
latent_ids = _prepare_latent_ids(latents)
# 生成 4D 坐标 (T, H, W, L)
# latent_ids: [1, 1024, 4]
# 每行形如 [0, h, w, 0]，其中 h in [0,32), w in [0,32)

# Step 4: 打包潜在表示
latents = _pack_latents(latents)
# [1, 128, 32, 32] -> [1, 1024, 128]
# 将空间维度展平到序列维度
```

**_prepare_latent_ids详解**:
```python
def _prepare_latent_ids(latents):
    """
    为每个潜在token生成4D位置坐标

    输入: latents [B, C, H, W]
    输出: latent_ids [B, H*W, 4]

    坐标含义:
    - T (dim 0): 时间维度，图像为0
    - H (dim 1): 高度索引 [0, H-1]
    - W (dim 2): 宽度索引 [0, W-1]
    - L (dim 3): 层级维度，单层为0
    """
    B, C, H, W = latents.shape  # [1, 128, 32, 32]

    t = torch.arange(1)         # [0]
    h = torch.arange(H)         # [0, 1, ..., 31]
    w = torch.arange(W)         # [0, 1, ..., 31]
    layer_ids = torch.arange(1) # [0]

    # 笛卡尔积: 所有可能的组合
    # [(t=0, h=0, w=0, l=0), (t=0, h=0, w=1, l=0), ...]
    latent_ids = torch.cartesian_prod(t, h, w, layer_ids)
    # latent_ids: [1024, 4]

    # 扩展到批大小
    latent_ids = latent_ids.unsqueeze(0).expand(B, -1, -1)
    # latent_ids: [1, 1024, 4]

    return latent_ids
```

---

### 1.4 `_encode_vae_image()`

**功能**: 使用VAE编码图像到潜在空间

**输入**:
```python
def _encode_vae_image(
    self,
    image: torch.Tensor,       # 输入图像 [B, 3, H, W]
    generator: torch.Generator, # 随机数生成器
) -> torch.Tensor:
```

**输入形状示例**:
```python
image = torch.randn(1, 3, 1024, 1024)  # RGB图像
```

**输出**:
```python
image_latents: torch.Tensor  # [B, 128, H/16, W/16]
```

**详细计算过程**:
```python
# Step 1: VAE编码
latents = vae.encode(image)
# 返回 VAE 的分布参数 (mean, logvar)

# Step 2: 采样 (使用argmax模式)
image_latents = retrieve_latents(latents, sample_mode="argmax")
# image_latents: [1, 32, 64, 64]

# Step 3: Patchify
image_latents = _patchify_latents(image_latents)
# 将 2x2 patches 合并到通道维度
# [1, 32, 64, 64] -> [1, 128, 32, 32]

# Step 4: BatchNorm归一化
bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)  # [1, 128, 1, 1]
bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + eps)
image_latents = (image_latents - bn_mean) / bn_std
```

---

## 2. Transformer层方法

### 2.1 `Flux2Transformer2DModel.forward()`

**功能**: Transformer核心前向传播，预测噪声

**输入**:
```python
def forward(
    self,
    hidden_states: torch.Tensor,           # 图像潜在表示
    encoder_hidden_states: torch.Tensor,   # 文本嵌入
    timestep: torch.LongTensor,            # 时间步
    img_ids: torch.Tensor,                 # 图像位置ID
    txt_ids: torch.Tensor,                 # 文本位置ID
    guidance: torch.Tensor | None,         # CFG引导值
) -> Transformer2DModelOutput:
```

**输入形状示例**:
```python
# 生成 1024x1024 图像
hidden_states: torch.Tensor      # [1, 1024, 512]  # 图像tokens
encoder_hidden_states: torch.Tensor  # [1, 512, 15360]  # 文本tokens
timestep: torch.Tensor           # [1]  # 例如 0.5
img_ids: torch.Tensor            # [1024, 4]  # 图像位置
txt_ids: torch.Tensor            # [512, 4]  # 文本位置
guidance: torch.Tensor           # [1]  # 例如 4.0
```

**输出**:
```python
output: Transformer2DModelOutput
output.sample: torch.Tensor      # [1, 1024, 512]  # 预测的噪声
```

**详细计算流程**:

```python
# ========== Step 1: 时间步嵌入 ==========
# timestep: [1] -> 标量值，如 0.5
timestep = timestep * 1000  # 缩放到 [0, 1000] 范围

# 时间投影
timesteps_proj = time_proj(timestep)  # [1, 256] 正弦位置编码
timesteps_emb = timestep_embedder(timesteps_proj)  # [1, 6144]

# 引导嵌入
if guidance is not None:
    guidance_proj = time_proj(guidance * 1000)  # [1, 256]
    guidance_emb = guidance_embedder(guidance_proj)  # [1, 6144]
    temb = timesteps_emb + guidance_emb  # [1, 6144]
else:
    temb = timesteps_emb

# ========== Step 2: 调制参数 ==========
# 图像调制
double_stream_mod_img = double_stream_modulation_img(temb)
# 返回 2 组参数: ((shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp))
# 每个参数: [1, 1, 6144]

# 文本调制
double_stream_mod_txt = double_stream_modulation_txt(temb)
# 同上结构

# 单流调制
single_stream_mod = single_stream_modulation(temb)
# 返回 1 组参数: ((shift, scale, gate),)
# 每个参数: [1, 1, 6144]

# ========== Step 3: 输入嵌入 ==========
hidden_states = x_embedder(hidden_states)
# [1, 1024, 512] -> [1, 1024, 6144]

encoder_hidden_states = context_embedder(encoder_hidden_states)
# [1, 512, 15360] -> [1, 512, 6144]

# ========== Step 4: 位置编码 ==========
txt_freqs_cos, txt_freqs_sin, img_freqs_cos, img_freqs_sin = \
    rope_prepare(img_ids, txt_ids)

# txt_freqs_cos: [512, 128]  # 文本RoPE余弦
# txt_freqs_sin: [512, 128]  # 文本RoPE正弦
# img_freqs_cos: [1024, 128] # 图像RoPE余弦
# img_freqs_sin: [1024, 128] # 图像RoPE正弦

# 合并
concat_freqs_cos = torch.cat([txt_freqs_cos, img_freqs_cos], dim=0)
concat_freqs_sin = torch.cat([txt_freqs_sin, img_freqs_sin], dim=0)
# concat_freqs_cos: [1536, 128]
# concat_freqs_sin: [1536, 128]

# ========== Step 5: Double-Stream Blocks ==========
for block in transformer_blocks:  # 8 blocks
    encoder_hidden_states, hidden_states = block(
        hidden_states=hidden_states,           # [1, 1024, 6144]
        encoder_hidden_states=encoder_hidden_states,  # [1, 512, 6144]
        temb_mod_params_img=double_stream_mod_img,
        temb_mod_params_txt=double_stream_mod_txt,
        image_rotary_emb=(concat_freqs_cos, concat_freqs_sin),
    )

# ========== Step 6: 合并文本和图像 ==========
hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
# hidden_states: [1, 1536, 6144]

# ========== Step 7: Single-Stream Blocks ==========
for block in single_transformer_blocks:  # 48 blocks
    hidden_states = block(
        hidden_states=hidden_states,  # [1, 1536, 6144]
        temb_mod_params=single_stream_mod,
        image_rotary_emb=(concat_freqs_cos, concat_freqs_sin),
        text_seq_len=512,
    )

# ========== Step 8: 输出 ==========
# 移除文本tokens
hidden_states = hidden_states[:, 512:]  # [1, 1024, 6144]

# 输出归一化
hidden_states = norm_out(hidden_states, temb)

# 投影到输出通道
output = proj_out(hidden_states)
# output: [1, 1024, 512]  # 与输入相同形状

return Transformer2DModelOutput(sample=output)
```

---

### 2.2 `Flux2TransformerBlock.forward()`

**功能**: 双流Transformer块，分离处理图像和文本

**输入**:
```python
def forward(
    self,
    hidden_states: torch.Tensor,           # 图像tokens [B, img_seq, D]
    encoder_hidden_states: torch.Tensor,   # 文本tokens [B, txt_seq, D]
    temb_mod_params_img: tuple,            # 图像调制参数
    temb_mod_params_txt: tuple,            # 文本调制参数
    image_rotary_emb: tuple,               # RoPE嵌入 (cos, sin)
) -> tuple[torch.Tensor, torch.Tensor]:
```

**输入形状示例**:
```python
hidden_states: [1, 1024, 6144]
encoder_hidden_states: [1, 512, 6144]
temb_mod_params_img = (
    (shift_msa, scale_msa, gate_msa),      # 每个参数: [1, 1, 6144]
    (shift_mlp, scale_mlp, gate_mlp),
)
image_rotary_emb = (freqs_cos, freqs_sin)  # freqs_cos: [1536, 128]
```

**输出**:
```python
encoder_hidden_states: torch.Tensor  # [1, 512, 6144]
hidden_states: torch.Tensor          # [1, 1024, 6144]
```

**详细计算流程**:
```python
# ========== Step 1: 解包调制参数 ==========
(shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
(c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

# ========== Step 2: 图像分支预处理 ==========
# LayerNorm
norm_hidden_states = norm1(hidden_states)  # [1, 1024, 6144]

# 调制: y = (1 + scale) * x + shift
norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

# ========== Step 3: 文本分支预处理 ==========
norm_encoder_hidden_states = norm1_context(encoder_hidden_states)
norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

# ========== Step 4: 联合注意力 ==========
attn_output, context_attn_output = attn(
    hidden_states=norm_hidden_states,            # [1, 1024, 6144]
    encoder_hidden_states=norm_encoder_hidden_states,  # [1, 512, 6144]
    image_rotary_emb=(freqs_cos, freqs_sin),
)
# attn_output: [1, 1024, 6144]
# context_attn_output: [1, 512, 6144]

# ========== Step 5: 图像分支残差 + FFN ==========
# 注意力残差
hidden_states = hidden_states + gate_msa * attn_output

# FFN
norm_hidden_states = norm2(hidden_states)
norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
ff_output = ff(norm_hidden_states)  # [1, 1024, 6144]
hidden_states = hidden_states + gate_mlp * ff_output

# ========== Step 6: 文本分支残差 + FFN ==========
# 注意力残差
encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output

# FFN
norm_encoder_hidden_states = norm2_context(encoder_hidden_states)
norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
context_ff_output = ff_context(norm_encoder_hidden_states)
encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output

return encoder_hidden_states, hidden_states
```

---

### 2.3 `Flux2SingleTransformerBlock.forward()`

**功能**: 单流Transformer块，融合处理图像和文本

**输入**:
```python
def forward(
    self,
    hidden_states: torch.Tensor,    # 合并的tokens [B, txt_seq+img_seq, D]
    temb_mod_params: tuple,         # 调制参数
    image_rotary_emb: tuple,        # RoPE嵌入
    text_seq_len: int,              # 文本序列长度
) -> torch.Tensor:
```

**输入形状示例**:
```python
hidden_states: [1, 1536, 6144]  # 512 文本 + 1024 图像
temb_mod_params = ((shift, scale, gate),)
image_rotary_emb = (freqs_cos, freqs_sin)
text_seq_len = 512
```

**输出**:
```python
hidden_states: torch.Tensor  # [1, 1536, 6144]
```

**详细计算流程**:
```python
# ========== Step 1: 解包调制参数 ==========
mod_shift, mod_scale, mod_gate = temb_mod_params[0]
# 每个参数: [1, 1, 6144]

# ========== Step 2: LayerNorm + 调制 ==========
norm_hidden_states = norm(hidden_states)  # [1, 1536, 6144]
norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

# ========== Step 3: 融合的 Attention + MLP ==========
# 一次投影同时生成 QKV 和 MLP 输入
hidden_states_proj, _ = to_qkv_mlp_proj(norm_hidden_states)
# hidden_states_proj: [1, 1536, 6144*3 + 24576] = [1, 1536, 43008]

# 分离 QKV 和 MLP
qkv, mlp_hidden_states = torch.split(
    hidden_states_proj,
    [3 * 6144, 24576],  # [18432, 24576]
    dim=-1
)
# qkv: [1, 1536, 18432]
# mlp_hidden_states: [1, 1536, 24576]

# 分离 Q, K, V
query, key, value = qkv.chunk(3, dim=-1)
# query, key, value: [1, 1536, 6144] each

# 重塑为多头
query = query.view(1, 1536, 48, 128)  # [B, seq, heads, head_dim]
key = key.view(1, 1536, 48, 128)
value = value.view(1, 1536, 48, 128)

# RMSNorm
query = norm_q(query)
key = norm_k(key)

# 应用 RoPE
query, key = rope(query, key, image_rotary_emb)

# 注意力计算
attn_output = attn(query, key, value)  # [1, 1536, 48, 128]
attn_output = attn_output.flatten(2, 3)  # [1, 1536, 6144]

# MLP 激活
mlp_hidden_states = mlp_act_fn(mlp_hidden_states)  # SwiGLU
# mlp_hidden_states: [1, 1536, 12288]  # 输出维度

# 合并注意力和 MLP 输出
hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
# hidden_states: [1, 1536, 18432]

# 输出投影
hidden_states, _ = to_out(hidden_states)
# hidden_states: [1, 1536, 6144]

# ========== Step 4: 残差连接 ==========
hidden_states = hidden_states + mod_gate * hidden_states

return hidden_states
```

---

## 3. 注意力机制方法

### 3.1 `Flux2Attention.forward()`

**功能**: 双流联合注意力

**输入**:
```python
def forward(
    self,
    hidden_states: torch.Tensor,           # 图像tokens [B, img_seq, D]
    encoder_hidden_states: torch.Tensor,   # 文本tokens [B, txt_seq, D]
    image_rotary_emb: tuple,               # (cos, sin)
) -> tuple[torch.Tensor, torch.Tensor]:
```

**输入形状示例**:
```python
hidden_states: [1, 1024, 6144]      # 图像
encoder_hidden_states: [1, 512, 6144]  # 文本
image_rotary_emb = (
    freqs_cos,  # [1536, 128]
    freqs_sin,  # [1536, 128]
)
```

**输出**:
```python
hidden_states: torch.Tensor          # [1, 1024, 6144] 图像输出
encoder_hidden_states: torch.Tensor  # [1, 512, 6144] 文本输出
```

**详细计算流程**:
```python
# ========== Step 1: 图像 QKV 投影 ==========
qkv, _ = to_qkv(hidden_states)
# qkv: [1, 1024, 18432]  # 3 * 6144

query, key, value = qkv.chunk(3, dim=-1)
# query, key, value: [1, 1024, 6144] each

# ========== Step 2: 文本 QKV 投影 ==========
encoder_qkv, _ = add_kv_proj(encoder_hidden_states)
# encoder_qkv: [1, 512, 18432]

encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)
# encoder_query, encoder_key, encoder_value: [1, 512, 6144] each

# ========== Step 3: 重塑为多头 + RMSNorm ==========
# 图像
query = query.view(1, 1024, 48, 128)  # [B, seq, heads, head_dim]
key = key.view(1, 1024, 48, 128)
value = value.view(1, 1024, 48, 128)
query = norm_q(query)
key = norm_k(key)

# 文本
encoder_query = encoder_query.view(1, 512, 48, 128)
encoder_key = encoder_key.view(1, 512, 48, 128)
encoder_value = encoder_value.view(1, 512, 48, 128)
encoder_query = norm_added_q(encoder_query)
encoder_key = norm_added_k(encoder_key)

# ========== Step 4: 合并 Q, K, V ==========
query = torch.cat([encoder_query, query], dim=1)
# query: [1, 1536, 48, 128]

key = torch.cat([encoder_key, key], dim=1)
# key: [1, 1536, 48, 128]

value = torch.cat([encoder_value, value], dim=1)
# value: [1, 1536, 48, 128]

# ========== Step 5: 应用 RoPE ==========
cos, sin = image_rotary_emb
cos = cos.to(query.dtype)  # [1536, 128]
sin = sin.to(query.dtype)  # [1536, 128]

query = rope(query, cos, sin)
key = rope(key, cos, sin)

# ========== Step 6: 注意力计算 ==========
# Flash Attention 内核
output = flash_attn_func(query, key, value)
# output: [1, 1536, 48, 128]

output = output.flatten(2, 3)
# output: [1, 1536, 6144]

# ========== Step 7: 分离输出 ==========
context_len = 512
encoder_hidden_states, hidden_states = output.split([context_len, output.shape[1] - context_len], dim=1)
# encoder_hidden_states: [1, 512, 6144]
# hidden_states: [1, 1024, 6144]

# ========== Step 8: 输出投影 ==========
hidden_states = to_out[0](hidden_states.contiguous())
encoder_hidden_states = to_add_out(encoder_hidden_states.contiguous())

return hidden_states, encoder_hidden_states
```

---

### 3.2 `Flux2ParallelSelfAttention.forward()`

**功能**: 融合的Attention + MLP（用于Single-Stream块）

**输入**:
```python
def forward(
    self,
    hidden_states: torch.Tensor,    # [B, seq, D]
    image_rotary_emb: tuple,        # (cos, sin)
) -> torch.Tensor:
```

**输入形状示例**:
```python
hidden_states: [1, 1536, 6144]
image_rotary_emb = (freqs_cos, freqs_sin)
```

**输出**:
```python
output: torch.Tensor  # [1, 1536, 6144]
```

**融合投影详解**:
```python
# 传统方法 (分开投影):
# QKV投影: hidden_states -> Q, K, V (3次矩阵乘法)
# MLP投影: hidden_states -> MLP_hidden (1次矩阵乘法)
# 总计: 4次矩阵乘法

# 融合方法:
# 一次投影: hidden_states -> [Q, K, V, MLP_hidden]
# 总计: 1次矩阵乘法 (3x更快)

# 投影矩阵
W_fused = torch.cat([W_q, W_k, W_v, W_mlp_in], dim=1)
# W_fused: [6144, 43008]  # 6144 * (3 + 4)

# 前向传播
output = hidden_states @ W_fused
# output: [1, 1536, 43008]

# 分离
qkv = output[:, :, :18432]       # Q, K, V
mlp_hidden = output[:, :, 18432:]  # MLP input
```

---

## 4. 位置编码方法

### 4.1 `Flux2PosEmbed.forward()`

**功能**: 计算4D旋转位置编码

**输入**:
```python
def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ids: [seq_len, 4] 或 [B, seq_len, 4]
    """
```

**输入形状示例**:
```python
# 图像位置ID
img_ids = torch.cartesian_prod(
    torch.tensor([0]),      # T
    torch.arange(32),       # H
    torch.arange(32),       # W
    torch.tensor([0]),      # L
)
# img_ids: [1024, 4]
# 例如: [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0], ...
```

**输出**:
```python
freqs_cos: torch.Tensor  # [seq_len, total_rotary_dim]
freqs_sin: torch.Tensor  # [seq_len, total_rotary_dim]
```

**详细计算**:
```python
theta = 2000  # RoPE基频
axes_dim = [32, 32, 32, 32]  # 每个维度的旋转维度

cos_out = []
sin_out = []

for i in range(4):  # 4个维度
    # 提取第i个维度的坐标
    pos = ids[:, i].float()  # [seq_len]

    # 计算频率
    dim = axes_dim[i]  # 32
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # freqs: [16]

    # 计算角度
    angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)
    # angles: [seq_len, 16]

    # 计算cos和sin
    cos_out.append(angles.cos())
    sin_out.append(angles.sin())

# 拼接所有维度
freqs_cos = torch.cat(cos_out, dim=-1)  # [seq_len, 128]
freqs_sin = torch.cat(sin_out, dim=-1)  # [seq_len, 128]
```

**RoPE应用示例**:
```python
def apply_rotary_emb(x, cos, sin):
    """
    x: [B, seq, heads, head_dim]
    cos, sin: [seq, head_dim/2]

    应用旋转位置编码
    """
    # 将x分成两半
    x1, x2 = x.chunk(2, dim=-1)

    # 旋转
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)

    return rotated
```

---

## 5. 完整推理示例

### 5.1 生成一张1024x1024图像

```python
import torch
from vllm_omni.diffusion.models.flux2_klein import Flux2KleinPipeline

# 初始化管道
pipeline = Flux2KleinPipeline(
    od_config=OmniDiffusionConfig(
        model="black-forest-labs/FLUX.2-Klein-4B",
        dtype=torch.bfloat16,
    )
)

# 准备输入
prompt = "A serene landscape with mountains and a lake at sunset"

# 推理
output = pipeline.forward(
    req=OmniDiffusionRequest(
        prompts=[prompt],
        sampling_params=SamplingParams(
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=4.0,
        ),
    )
)

# 输出
image = output.output  # PIL.Image
image.save("output.png")
```

### 5.2 详细张量形状追踪

```python
# ========== 初始化阶段 ==========
# 文本输入
prompt = "A cat"
input_ids: [1, 512]  # 分词后，padding到512

# 文本编码
# Qwen3ForCausalLM 前向传播
hidden_state_9:  [1, 512, 5120]
hidden_state_18: [1, 512, 5120]
hidden_state_27: [1, 512, 5120]

# 堆叠并重塑
stacked: [1, 3, 512, 5120]
permuted: [1, 512, 3, 5120]
prompt_embeds: [1, 512, 15360]

# 文本位置ID
text_ids: [1, 512, 4]
# 每行: [0, 0, 0, seq_pos]

# ========== 潜在空间初始化 ==========
# 目标尺寸: 1024x1024
# VAE下采样: 16x
# Patchify: 2x
# 最终潜在尺寸: 1024/32*2 = 64

# 噪声生成
noise: [1, 512, 32, 32]  # 512 = 128*4 channels (patchified)

# 打包
latents: [1, 1024, 512]  # 32*32 = 1024 tokens

# 潜在位置ID
latent_ids: [1, 1024, 4]
# 每行: [0, h, w, 0], h in [0,32), w in [0,32)

# ========== 时间步准备 ==========
num_steps = 50
timesteps: [50]  # 50个时间步

# ========== 去噪循环 (单步详解) ==========
t = timesteps[0]  # 例如 t=0.98

# 时间步嵌入
timestep = torch.tensor([t * 1000])  # [1]
temb: [1, 6144]

# 调制参数
double_mod_img = ((shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp))
# 每个参数: [1, 1, 6144]

# Transformer前向传播
# 输入嵌入
hidden_states: [1, 1024, 512] -> x_embedder -> [1, 1024, 6144]
encoder_hidden_states: [1, 512, 15360] -> context_embedder -> [1, 512, 6144]

# RoPE
txt_freqs_cos, txt_freqs_sin: [512, 128]
img_freqs_cos, img_freqs_sin: [1024, 128]
concat_freqs_cos, concat_freqs_sin: [1536, 128]

# Double-Stream Blocks (x8)
for block in transformer_blocks:
    # 图像预处理
    norm_img: [1, 1024, 6144]
    modulated_img = (1 + scale) * norm_img + shift

    # 文本预处理
    norm_txt: [1, 512, 6144]
    modulated_txt = (1 + c_scale) * norm_txt + c_shift

    # 联合注意力
    Q_img: [1, 1024, 48, 128]
    K_img: [1, 1024, 48, 128]
    V_img: [1, 1024, 48, 128]
    Q_txt: [1, 512, 48, 128]
    K_txt: [1, 512, 48, 128]
    V_txt: [1, 512, 48, 128]

    # 合并
    Q: [1, 1536, 48, 128]
    K: [1, 1536, 48, 128]
    V: [1, 1536, 48, 128]

    # 注意力
    scores = Q @ K.T / sqrt(128)  # [1, 48, 1536, 1536]
    attn = softmax(scores) @ V    # [1, 1536, 48, 128]

    # 分离
    txt_out: [1, 512, 6144]
    img_out: [1, 1024, 6144]

    # FFN
    img_out = img_out + gate * FFN(img_out)
    txt_out = txt_out + c_gate * FFN(txt_out)

# 合并
hidden_states = concat([encoder_hidden_states, hidden_states])
# hidden_states: [1, 1536, 6144]

# Single-Stream Blocks (x48)
for block in single_transformer_blocks:
    # 融合投影
    fused: [1, 1536, 43008]  # QKV + MLP
    qkv, mlp = split(fused)
    # qkv: [1, 1536, 18432]
    # mlp: [1, 1536, 24576]

    # 注意力 + MLP
    attn_out: [1, 1536, 6144]
    mlp_out: [1, 1536, 12288]

    # 合并 + 投影
    output: [1, 1536, 6144]

# 输出
hidden_states = hidden_states[:, 512:]  # 移除文本
# hidden_states: [1, 1024, 6144]
output = proj_out(hidden_states)
# noise_pred: [1, 1024, 512]

# Scheduler步骤
latents = latents + dt * noise_pred
# latents: [1, 1024, 512]

# ========== VAE解码 ==========
# 解包
latents: [1, 1024, 512] -> unpack -> [1, 512, 32, 32]

# Unpatchify
latents: [1, 512, 32, 32] -> unpatchify -> [1, 128, 64, 64]

# BatchNorm逆变换
latents = latents * bn_std + bn_mean

# VAE解码
image = vae.decode(latents)
# image: [1, 3, 1024, 1024]

# 后处理
image = image.clip(0, 1)
image = (image * 255).to(torch.uint8)
image = PIL.Image.fromarray(image)
```

### 5.3 CFG并行示例

```python
# 配置
cfg_parallel_size = 2  # 使用2个GPU并行计算CFG

# GPU 0: 条件输入
positive_prompt = "A beautiful landscape"
positive_embeds: [1, 512, 15360]

# GPU 1: 无条件输入
negative_prompt = ""
negative_embeds: [1, 512, 15360]

# 并行计算
# GPU 0
noise_pred_cond = transformer(latents, positive_embeds, t)

# GPU 1
noise_pred_uncond = transformer(latents, negative_embeds, t)

# AllReduce合并
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
```

---

## 附录: 关键参数速查表

| 组件 | 参数名 | 形状 | 说明 |
|------|--------|------|------|
| Text Encoder | input_ids | [B, 512] | 分词后的token IDs |
| Text Encoder | prompt_embeds | [B, 512, 15360] | 文本嵌入 |
| Text Encoder | text_ids | [B, 512, 4] | 文本位置坐标 |
| Latents | latents | [B, H*W/64, 512] | 打包后的潜在表示 |
| Latents | latent_ids | [B, H*W/64, 4] | 潜在位置坐标 |
| Transformer | hidden_states | [B, seq, 6144] | 隐藏状态 |
| Transformer | temb | [B, 6144] | 时间步嵌入 |
| Attention | Q/K/V | [B, seq, 48, 128] | 查询/键/值 |
| Attention | freqs_cos/sin | [seq, 128] | RoPE嵌入 |
| VAE | image | [B, 3, H, W] | 输出图像 |
