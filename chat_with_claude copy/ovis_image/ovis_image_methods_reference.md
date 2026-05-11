# OvisImage 模型方法参考手册

## 1. Pipeline 类方法详解

### OvisImagePipeline

**文件**: `vllm_omni/diffusion/models/ovis_image/pipeline_ovis_image.py`

#### 构造函数

```python
def __init__(
    self,
    *,
    od_config: OmniDiffusionConfig,
    prefix: str = "",
):
```

**初始化组件**:
- `scheduler`: FlowMatchEulerDiscreteScheduler
- `text_encoder`: Qwen3Model
- `vae`: AutoencoderKL
- `tokenizer`: Qwen2TokenizerFast
- `transformer`: OvisImageTransformer2DModel

---

#### forward 方法 (主入口)

```python
def forward(
    self,
    req: OmniDiffusionRequest,
    prompt: str | list[str] | None = None,
    negative_prompt: str | list[str] | None = None,
    guidance_scale: float = 5.0,
    height: int | None = None,
    width: int | None = None,
    num_inference_steps: int = 50,
    sigmas: list[float] | None = None,
    num_images_per_prompt: int | None = 1,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.FloatTensor | None = None,
    prompt_embeds: torch.FloatTensor | None = None,
    negative_prompt_embeds: torch.FloatTensor | None = None,
    output_type: str | None = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: dict[str, Any] | None = None,
    callback_on_step_end: Callable[[int, int, dict], None] | None = None,
    callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    max_sequence_length: int = 256,
) -> DiffusionOutput:
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| req | OmniDiffusionRequest | - | 请求对象，包含 prompts 和 sampling_params |
| prompt | str \| List[str] | None | 文本提示词 |
| negative_prompt | str \| List[str] | None | 负向提示词 |
| guidance_scale | float | 5.0 | CFG 缩放因子 |
| height | int | 1024 | 图像高度 |
| width | int | 1024 | 图像宽度 |
| num_inference_steps | int | 50 | 去噪步数 |
| sigmas | List[float] | None | 自定义噪声调度 |
| num_images_per_prompt | int | 1 | 每个提示生成的图像数 |
| generator | torch.Generator | None | 随机数生成器 |
| latents | Tensor | None | 预生成的 latents |
| prompt_embeds | Tensor | None | 预计算的文本嵌入 |
| negative_prompt_embeds | Tensor | None | 预计算的负向嵌入 |
| output_type | str | "pil" | 输出格式 ("pil", "latent", "pt") |
| max_sequence_length | int | 256 | 文本最大序列长度 |

**返回值**:
- `DiffusionOutput`: 包含 `output` (图像张量) 和 `stage_durations` (性能统计)

---

#### encode_prompt 方法

```python
def encode_prompt(
    self,
    prompt: str | list[str],
    device: torch.device | None = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: torch.FloatTensor | None = None,
):
```

**功能**: 将文本提示编码为嵌入向量

**处理流程**:
1. 调用 `_get_ovis_prompt_embeds` 进行编码
2. 生成文本位置 ID (`text_ids`)

**返回值**:
- `prompt_embeds`: [B, seq_len, 2048]
- `text_ids`: [seq_len, 3]

---

#### _get_ovis_prompt_embeds 方法

```python
def _get_ovis_prompt_embeds(
    self,
    prompt: str | list[str] = None,
    num_images_per_prompt: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
):
```

**功能**: 内部方法，实际执行文本编码

**关键步骤**:
```python
# 1. 构建 messages
messages = self._get_messages(prompt)

# 2. 分词
tokens = self.tokenizer(
    messages,
    padding="max_length",
    truncation=True,
    max_length=self.tokenizer_max_length,
    return_tensors="pt",
    add_special_tokens=False,
)

# 3. 编码
outputs = self.text_encoder(
    input_ids=input_ids,
    attention_mask=attention_mask,
)

# 4. 后处理
prompt_embeds = outputs.last_hidden_state
prompt_embeds = prompt_embeds * attention_mask[..., None]
prompt_embeds = prompt_embeds[:, self.user_prompt_begin_id:, :]  # 截取有效部分
```

---

#### prepare_latents 方法

```python
def prepare_latents(
    self,
    batch_size,
    num_channel_latents,
    height,
    width,
    dtype,
    device,
    generator,
    latents=None,
):
```

**功能**: 初始化噪声 latents

**计算过程**:
```python
# 1. 计算 latent 尺寸 (缩小 16 倍)
height = int(2 * (int(height) // (self.vae_scale_factor * 2)))  # H/16
width = int(2 * (int(width) // (self.vae_scale_factor * 2)))    # W/16

# 2. 生成噪声
shape = (batch_size, num_channel_latents, height, width)
latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

# 3. Pack latents
latents = self._pack_latents(latents, batch_size, num_channel_latents, height, width)
# 形状: [B, (H/16 * W/16) / 4, 64]

# 4. 生成位置 ID
latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
# 形状: [(H/16 * W/16) / 4, 3]
```

**返回值**:
- `latents`: [B, img_seq, 64]
- `latent_image_ids`: [img_seq, 3]

---

#### diffuse 方法

```python
def diffuse(
    self,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    negative_text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    do_true_cfg: bool,
    guidance_scale: float,
    cfg_normalize: bool = False,
) -> torch.Tensor:
```

**功能**: 执行扩散去噪循环

**核心循环**:
```python
for i, t in enumerate(timesteps):
    timestep = t.expand(latents.shape[0]).to(latents.dtype)

    # 构建正向条件输入
    positive_kwargs = {
        "hidden_states": latents,
        "timestep": timestep / 1000,
        "encoder_hidden_states": prompt_embeds,
        "txt_ids": text_ids,
        "img_ids": latent_image_ids,
        "return_dict": False,
    }

    # 预测噪声 (自动处理 CFG)
    noise_pred = self.predict_noise_maybe_with_cfg(
        do_true_cfg, guidance_scale, positive_kwargs, negative_kwargs, cfg_normalize
    )

    # Scheduler 更新
    latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
```

---

#### 辅助方法

**_pack_latents**: 2D→1D 打包
```python
@staticmethod
def _pack_latents(latents, batch_size, num_channel_latents, height, width):
    # [B, C, H, W] → [B, (H/2)*(W/2), C*4]
    latents = latents.view(batch_size, num_channel_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channel_latents * 4)
    return latents
```

**_unpack_latents**: 1D→2D 解包
```python
@staticmethod
def _unpack_latents(latents, height, width, vae_scale_factor):
    # [B, seq, 64] → [B, 16, H/16, W/16]
    height = int(2 * (int(height) // (vae_scale_factor * 2)))
    width = int(2 * (int(width) // (vae_scale_factor * 2)))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    return latents
```

**_prepare_latent_image_ids**: 生成图像位置 ID
```python
@staticmethod
def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    # 生成 (h, w) 网格位置
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]  # y
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]   # x
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)
```

---

## 2. Transformer 类方法详解

### OvisImageTransformer2DModel

**文件**: `vllm_omni/diffusion/models/ovis_image/ovis_image_transformer.py`

#### 构造函数

```python
def __init__(
    self,
    od_config: OmniDiffusionConfig,
    patch_size: int = 1,
    in_channels: int = 64,
    out_channels: int | None = 64,
    num_layers: int = 6,
    num_single_layers: int = 27,
    attention_head_dim: int = 128,
    num_attention_heads: int = 24,
    joint_attention_dim: int = 2048,
    axes_dims_rope: tuple[int] = (16, 56, 56),
):
```

**构建的模块**:
```python
self.pos_embed = OvisImagePosEmbed(theta=10000, axes_dim=axes_dims_rope)
self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)
self.context_embedder_norm = RMSNorm(joint_attention_dim, eps=1e-6)
self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
self.x_embedder = nn.Linear(in_channels, self.inner_dim)
self.transformer_blocks = nn.ModuleList([...])  # 6 个双流块
self.single_transformer_blocks = nn.ModuleList([...])  # 27 个单流块
self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
```

---

#### forward 方法

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    return_dict: bool = True,
) -> torch.Tensor | Transformer2DModelOutput:
```

**参数**:

| 参数 | 形状 | 说明 |
|------|------|------|
| hidden_states | [B, img_seq, 64] | 图像 latents |
| encoder_hidden_states | [B, txt_seq, 2048] | 文本嵌入 |
| timestep | [B] | 时间步 |
| img_ids | [img_seq, 3] | 图像位置 ID |
| txt_ids | [txt_seq, 3] | 文本位置 ID |

**返回值**:
- `Transformer2DModelOutput(sample=[B, img_seq, 64])`

---

#### load_weights 方法

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

**功能**: 加载预训练权重

**Stacked Parameters Mapping**:
```python
stacked_params_mapping = [
    (".to_qkv", ".to_q", "q"),  # QKV 合并
    (".to_qkv", ".to_k", "k"),
    (".to_qkv", ".to_v", "v"),
    (".add_kv_proj", ".add_q_proj", "q"),  # 编码器 QKV 合并
    (".add_kv_proj", ".add_k_proj", "k"),
    (".add_kv_proj", ".add_v_proj", "v"),
]
```

---

### OvisImageTransformerBlock

#### 构造函数

```python
def __init__(
    self,
    dim: int,
    num_attention_heads: int,
    attention_head_dim: int,
    qk_norm: str = "rms_norm",
    eps: float = 1e-6,
):
```

#### forward 方法

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

**处理流程**:
1. 对图像和文本分别进行 AdaLayerNormZero
2. 执行联合注意力 (图像+文本)
3. 对图像分支进行 FFN 处理
4. 对文本分支进行 FFN 处理
5. 返回更新后的文本和图像嵌入

---

### OvisImageSingleTransformerBlock

#### 构造函数

```python
def __init__(
    self,
    dim: int,
    num_attention_heads: int,
    attention_head_dim: int,
    mlp_ratio: float = 4.0
):
```

#### forward 方法

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

**处理流程**:
1. 拼接文本和图像嵌入
2. AdaLayerNormZeroSingle 归一化
3. 并行执行注意力和 MLP
4. 门控合并
5. 分离文本和图像嵌入

---

### OvisImageAttention

#### 构造函数

```python
def __init__(
    self,
    query_dim: int,
    heads: int = 8,
    dim_head: int = 64,
    dropout: float = 0.0,
    bias: bool = False,
    added_kv_proj_dim: int | None = None,
    added_proj_bias: bool | None = True,
    out_bias: bool = True,
    eps: float = 1e-5,
    out_dim: int = None,
    context_pre_only: bool | None = None,
    pre_only: bool = False,
):
```

**关键参数**:
- `pre_only=True`: 只计算注意力输出，不进行输出投影 (用于 SingleTransformerBlock)
- `added_kv_proj_dim`: 文本编码器投影维度

#### forward 方法

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    image_rotary_emb: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
```

---

### OvisImagePosEmbed

#### forward 方法

```python
def forward(self, ids: torch.Tensor) -> torch.Tensor:
    """
    Args:
        ids: [seq_len, 3] 位置 ID (t, h, w)

    Returns:
        (freqs_cos, freqs_sin): 各 [seq_len, 128]
    """
```

**实现细节**:
```python
for i in range(n_axes):  # 3 个轴
    freqs_cis = get_1d_rotary_pos_embed(
        self.axes_dim[i],  # 16, 56, 56
        pos[:, i],
        theta=self.theta,  # 10000
        use_real=False,
    )
    cos_out.append(freqs_cis.real)
    sin_out.append(freqs_cis.imag)

freqs_cos = torch.cat(cos_out, dim=-1)  # [seq_len, 128]
freqs_sin = torch.cat(sin_out, dim=-1)  # [seq_len, 128]
```

---

## 3. 辅助函数

### calculate_shift

```python
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
```

**功能**: 根据图像序列长度计算时间步 shift 参数

**公式**: `mu = m * image_seq_len + b`
- `m = (max_shift - base_shift) / (max_seq_len - base_seq_len)`
- `b = base_shift - m * base_seq_len`

---

### retrieve_timesteps

```python
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
```

**功能**: 从 scheduler 获取时间步序列

---

### get_ovis_image_post_process_func

```python
def get_ovis_image_post_process_func(
    od_config: OmniDiffusionConfig,
):
```

**功能**: 创建后处理函数，将 tensor 转换为 PIL 图像

---

## 4. 类型定义

### DiffusionOutput

```python
@dataclass
class DiffusionOutput:
    output: torch.Tensor  # 生成的图像 [B, 3, H, W]
    stage_durations: dict[str, float] | None  # 各阶段耗时
```

### OmniDiffusionRequest

```python
class OmniDiffusionRequest:
    prompts: list[str | dict]  # 提示词列表
    sampling_params: SamplingParams  # 采样参数
```

---

## 5. 配置参数完整列表

### Transformer 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| patch_size | 1 | Patch 大小 |
| in_channels | 64 | 输入通道数 |
| out_channels | 64 | 输出通道数 |
| num_layers | 6 | 双流 DiT 层数 |
| num_single_layers | 27 | 单流 DiT 层数 |
| attention_head_dim | 128 | 每个注意力头的维度 |
| num_attention_heads | 24 | 注意力头数量 |
| joint_attention_dim | 2048 | 文本嵌入维度 |
| axes_dims_rope | (16, 56, 56) | RoPE 各轴维度 |

### 采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| height | 1024 | 图像高度 |
| width | 1024 | 图像宽度 |
| num_inference_steps | 50 | 去噪步数 |
| guidance_scale | 5.0 | CFG 缩放因子 |
| num_images_per_prompt | 1 | 每提示生成图像数 |
| max_sequence_length | 256 | 文本最大长度 |

### Scheduler 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| base_image_seq_len | 256 | 基准序列长度 |
| max_image_seq_len | 4096 | 最大序列长度 |
| base_shift | 0.5 | 基准 shift |
| max_shift | 1.15 | 最大 shift |

---

## 6. 使用示例

### 基本使用

```python
from vllm_omni.diffusion.models.ovis_image import OvisImagePipeline

# 初始化
pipeline = OvisImagePipeline(od_config=config)

# 生成图像
output = pipeline(
    req=request,
    prompt="a beautiful sunset over mountains",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=5.0,
)

# 获取图像
image = output.output  # [B, 3, H, W] tensor
```

### 使用负向提示

```python
output = pipeline(
    req=request,
    prompt="a cat",
    negative_prompt="blurry, low quality",
    guidance_scale=7.5,
)
```

### 自定义随机种子

```python
generator = torch.Generator(device="cuda").manual_seed(42)
output = pipeline(
    req=request,
    prompt="a dog",
    generator=generator,
)
```
