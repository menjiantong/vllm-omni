# 配置参数详解

## 1. 配置类层次结构

```
Mammothmoda2Config (顶层配置)
│
├── llm_config: Mammothmoda2Qwen2_5_VLConfig
│   │
│   ├── text_config: Mammothmoda2Qwen2_5_VLTextConfig
│   │   ├── vocab_size: 152064
│   │   ├── hidden_size: 8192
│   │   ├── num_hidden_layers: 80
│   │   ├── num_attention_heads: 64
│   │   ├── extra_gen_vocab: True
│   │   ├── gen_vocab_size: 32800
│   │   └── moe_type: "ffn"
│   │
│   └── vision_config: Mammothmoda2Qwen2_5_VLVisionConfig
│       ├── depth: 32
│       ├── hidden_size: 3584
│       ├── num_heads: 16
│       └── patch_size: 14
│
├── gen_vae_config: dict (diffusers config)
│   └── AutoencoderKL 配置
│
├── gen_dit_config: dict (diffusers config)
│   └── Transformer2DModel 配置
│
├── gen_condition_mode: "text" | "image" | "text_image"
├── gen_image_condition_refiner_config: dict | None
├── gen_axes_dim_rope: [40, 40, 40]
└── gen_axes_lens: [10000, 10000, 10000]
```

## 2. Mammothmoda2Config

**位置**: `configs/mammoth_moda2.py:210-289`

```python
class Mammothmoda2Config(PretrainedConfig):
    model_type = "mammothmoda2"
    is_composition = True
    
    def __init__(
        self,
        *,
        # LLM 配置 (必须)
        llm_config: dict | None = None,
        
        # DiT/VAE 配置 (T2I 任务必须)
        gen_vae_config: dict | None = None,
        gen_dit_config: dict | None = None,
        
        # 条件模式
        gen_condition_mode: Literal["text", "image", "text_image"] = "image",
        
        # 图像条件精炼器配置 (可选)
        gen_image_condition_refiner_config: dict | None = None,
        
        # RoPE 配置
        gen_axes_dim_rope: list[int] | None = None,  # 默认 [40, 40, 40]
        gen_axes_lens: list[int] | None = None,      # 默认 [10000, 10000, 10000]
        
        # 其他
        gen_transport_config: dict | None = None,
        initializer_range: float = 0.02,
        architectures: list[str] | None = None,
        **kwargs,
    ):
```

### 2.1 关键代理属性

为了兼容 vLLM 的多模态处理，顶层配置代理了 `llm_config` 的属性:

```python
@property
def vision_config(self):
    return self.llm_config.vision_config

@property
def image_token_id(self) -> int:
    return self.llm_config.image_token_id

@property
def video_token_id(self) -> int:
    return self.llm_config.video_token_id

@property
def vision_start_token_id(self) -> int:
    return self.llm_config.vision_start_token_id

@property
def vision_end_token_id(self) -> int:
    return self.llm_config.vision_end_token_id
```

### 2.2 get_text_config()

```python
def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
    # 返回嵌套的 text_config，用于 vLLM 的采样验证
    return self.llm_config.text_config
```

---

## 3. Mammothmoda2Qwen2_5_VLTextConfig

**位置**: `configs/mammoth_moda2.py:61-139`

```python
class Mammothmoda2Qwen2_5_VLTextConfig(Qwen2_5_VLTextConfig):
    def __init__(
        self,
        # 标准 Qwen2.5 参数
        vocab_size: int = 152064,
        hidden_size: int = 8192,
        intermediate_size: int = 29568,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-05,
        rope_theta: float = 1000000.0,
        
        # MammothModa2 特有参数
        extra_gen_vocab: bool = True,         # 是否启用额外生成词汇表
        gen_vocab_size: int = 32800,          # 生成词汇表大小
        gen_vocab_start_index: int | None = None,  # 生成 token 起始 ID
        moe_type: str = "ffn",                # MoE 类型
        
        # 多模态 token ID
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        **kwargs,
    ):
```

### 3.1 词汇表计算

```python
# 如果启用 extra_gen_vocab
if extra_gen_vocab:
    if gen_vocab_start_index is None:
        gen_vocab_start_index = vocab_size  # 默认从 152064 开始
    
    # 扩展 vocab_size 以覆盖生成词汇
    vocab_size = gen_vocab_start_index + gen_vocab_size
    # 最终 vocab_size = 152064 + 32800 = 184864
```

### 3.2 MoE 类型说明

| moe_type | 含义 | 使用层 |
|----------|------|--------|
| `"none"` | 不启用 MoE | - |
| `"attention"` | 注意力层 MoE | 所有 attention 层 |
| `"ffn"` | FFN 层 MoE | 所有 FFN 层 |
| `"ffn_attention"` | 全部 MoE | 所有层 |
| `"ffn_attention-14:28"` | 部分层 MoE | 第 14-27 层 |

---

## 4. Mammothmoda2Qwen2_5_VLVisionConfig

**位置**: `configs/mammoth_moda2.py:20-58`

```python
class Mammothmoda2Qwen2_5_VLVisionConfig(Qwen2_5_VLVisionConfig):
    def __init__(
        self,
        depth: int = 32,               # ViT 层数
        hidden_size: int = 3584,       # 隐藏层维度
        hidden_act: str = "silu",
        intermediate_size: int = 3420,
        num_heads: int = 16,           # 注意力头数
        in_channels: int = 3,          # 输入图像通道
        patch_size: int = 14,          # Patch 大小
        spatial_merge_size: int = 2,   # 空间合并大小
        temporal_patch_size: int = 2,  # 时间 Patch 大小
        tokens_per_second: int = 4,
        window_size: int = 112,        # 窗口注意力大小
        out_hidden_size: int = 3584,   # 输出隐藏层维度
        fullatt_block_indexes: list[int] = [7, 15, 23, 31],  # 全注意力层索引
        initializer_range: float = 0.02,
        **kwargs,
    ):
```

---

## 5. Transformer2DModel 配置 (gen_dit_config)

```python
gen_dit_config = {
    "patch_size": 2,
    "in_channels": 16,           # Latent 通道数
    "out_channels": 16,
    "hidden_size": 2304,
    "num_layers": 26,            # 主干 Transformer 层数
    "num_refiner_layers": 2,     # 精炼器层数
    "num_attention_heads": 24,
    "num_kv_heads": 8,           # GQA
    "multiple_of": 256,          # FFN 维度对齐
    "ffn_dim_multiplier": None,
    "norm_eps": 1e-5,
    "axes_dim_rope": (32, 32, 32),
    "axes_lens": (300, 512, 512),
    "text_feat_dim": 1024,
    "timestep_scale": 1.0,
}
```

### 5.1 维度约束

```
hidden_size // num_attention_heads == sum(axes_dim_rope)
2304 // 24 == 96 == 32 + 32 + 32 ✓
```

---

## 6. AutoencoderKL 配置 (gen_vae_config)

```python
gen_vae_config = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "scaling_factor": 0.18215,    # Latent 缩放因子
    "shift_factor": None,         # Latent 偏移因子
}
```

---

## 7. Stage 配置文件

### 7.1 mammoth_moda2.yaml (T2I 两阶段)

```yaml
# vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml

stages:
  - name: ar
    model_stage: ar
    architectures:
      - MammothModa2ARForConditionalGeneration
    
  - name: dit
    model_stage: dit
    architectures:
      - MammothModa2DiTPipeline
    stage_input_processor: ar2dit
```

### 7.2 mammoth_moda2_ar.yaml (单阶段 AR)

```yaml
# vllm_omni/model_executor/stage_configs/mammoth_moda2_ar.yaml

stages:
  - name: ar
    model_stage: ar
    architectures:
      - MammothModa2ARForConditionalGeneration
```

---

## 8. 运行时参数

### 8.1 T2I 任务参数

```python
additional_information = {
    # 元数据
    "meta": {
        "omni_task": ["t2i"],
        "ar_width": 32,
        "ar_height": 32,
        "eol_token_id": 184862,
        "visual_token_start_id": 152064,
        "visual_token_end_id": 184863,
    },
    
    # 图像尺寸
    "image_height": [512],
    "image_width": [512],
    
    # 引导参数
    "text_guidance_scale": [7.5],
    "cfg_range": [0.0, 1.0],
    
    # 扩散步数
    "num_inference_steps": [20],
}
```

### 8.2 Understanding 任务参数

```python
# Understanding 任务不需要额外参数
# omni_task != "t2i" 时自动禁用 gen vocab
```

---

## 9. 特殊 Token ID

| Token | ID | 用途 |
|-------|-----|------|
| `<|vision_start|>` | 151652 | 视觉输入开始 |
| `<|vision_end|>` | 151653 | 视觉输入结束 |
| `<|image_pad|>` | 151655 | 图像填充 |
| `<|video_pad|>` | 151656 | 视频填充 |
| Base vocab end | 152063 | 基础词汇表结束 |
| Gen vocab start | 152064 | 生成词汇表开始 |
| `<|endofline|>` | 184862 | AR grid 行尾 |
| `<|endoffile|>` | 184863 | AR grid 结束 |
| Gen vocab end | 184863 | 生成词汇表结束 |

---

## 10. 配置示例

### 10.1 完整模型配置示例

```json
{
  "model_type": "mammothmoda2",
  "architectures": ["Mammothmoda2Model"],
  "tokenizer_class": "MammothUTokenizer",
  
  "llm_config": {
    "model_type": "mammothmoda2_qwen2_5_vl",
    "text_config": {
      "model_type": "mammothmoda2_qwen2_5_vl_text",
      "vocab_size": 152064,
      "hidden_size": 8192,
      "intermediate_size": 29568,
      "num_hidden_layers": 80,
      "num_attention_heads": 64,
      "num_key_value_heads": 8,
      "extra_gen_vocab": true,
      "gen_vocab_size": 32800,
      "moe_type": "ffn"
    },
    "vision_config": {
      "model_type": "mammothmoda2_qwen2_5_vl_vision",
      "depth": 32,
      "hidden_size": 3584,
      "num_heads": 16,
      "patch_size": 14
    },
    "image_token_id": 151655,
    "video_token_id": 151656
  },
  
  "gen_dit_config": {
    "patch_size": 2,
    "in_channels": 16,
    "hidden_size": 2304,
    "num_layers": 26,
    "num_attention_heads": 24,
    "num_kv_heads": 8,
    "axes_dim_rope": [32, 32, 32]
  },
  
  "gen_vae_config": {
    "latent_channels": 16,
    "scaling_factor": 0.18215
  },
  
  "gen_condition_mode": "image",
  "gen_axes_dim_rope": [40, 40, 40],
  "gen_axes_lens": [10000, 10000, 10000]
}
```
