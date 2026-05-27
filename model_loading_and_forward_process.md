# OvisImage 模型加载和前向传播过程详解

## 概述

本文档详细描述了 OvisImage 模型在 vLLM-Omni 框架中的加载和前向传播过程。所有关键步骤都已添加日志，日志前缀为 `----my_debug----`，方便定位。

---

## 一、模型加载流程

### 1. 整体架构

模型加载的调用链如下：

```
DiffusionWorker.__init__
    └── DiffusionWorker.init_device()          # 初始化设备和分布式环境
    └── DiffusionModelRunner.__init__()
    └── DiffusionWorker.load_model()
        └── DiffusionModelRunner.load_model()
            └── DiffusersPipelineLoader.load_model()
                └── initialize_model()          # 从注册表加载模型类
                    └── OvisImagePipeline.__init__()
                        ├── FlowMatchEulerDiscreteScheduler.from_pretrained()  # 加载 scheduler
                        ├── Qwen3Model.from_pretrained()                       # 加载 text_encoder
                        ├── AutoencoderKL.from_pretrained()                    # 加载 VAE
                        ├── Qwen2TokenizerFast.from_pretrained()               # 加载 tokenizer
                        └── OvisImageTransformer2DModel.__init__()             # 初始化 transformer
                └── DiffusersPipelineLoader.load_weights()
                    └── OvisImagePipeline.load_weights()
                        └── OvisImageTransformer2DModel.load_weights()         # 加载 transformer 权重
```

### 2. 详细步骤

#### 2.1 DiffusionWorker 初始化 (`diffusion_worker.py`)

```python
# 关键日志点：
----my_debug---- [DiffusionWorker.__init__] local_rank=0, rank=0
----my_debug---- [DiffusionWorker.__init__] od_config.model=<模型路径>
----my_debug---- [DiffusionWorker.__init__] od_config.model_class_name=OvisImagePipeline
```

主要工作：
- 设置分布式环境变量 (MASTER_ADDR, MASTER_PORT, LOCAL_RANK, RANK, WORLD_SIZE)
- 初始化设备 (CUDA device)
- 创建 vllm_config 配置并行参数
- 初始化分布式环境 (NCCL)
- 初始化模型并行组

#### 2.2 DiffusionModelRunner 初始化 (`diffusion_model_runner.py`)

```python
# 关键日志点：
----my_debug---- [DiffusionModelRunner.__init__] device=cuda:0
----my_debug---- [DiffusionModelRunner.__init__] od_config.dtype=torch.bfloat16
```

主要工作：
- 存储 od_config 和 device
- 初始化 KV transfer manager

#### 2.3 模型加载入口 (`diffusers_loader.py`)

```python
# 关键日志点：
----my_debug---- [load_model] od_config.model_class_name=OvisImagePipeline
----my_debug---- [load_model] load_device=cuda:0
----my_debug---- [load_model] target_device=cuda:0
```

主要工作：
- 设置默认 torch dtype
- 调用 initialize_model() 实例化模型
- 调用 load_weights() 加载权重

#### 2.4 模型注册和实例化 (`registry.py`)

```python
# 关键日志点：
----my_debug---- [initialize_model] od_config.model_class_name=OvisImagePipeline
----my_debug---- [initialize_model] model_class=<class 'OvisImagePipeline'>
```

模型注册表映射：
```python
_DIFFUSION_MODELS = {
    "OvisImagePipeline": (
        "ovis_image",           # mod_folder
        "pipeline_ovis_image",  # mod_relname
        "OvisImagePipeline",    # cls_name
    ),
}
```

主要工作：
- 从注册表加载模型类
- 准备量化配置
- 实例化模型
- 配置 VAE 内存优化 (use_slicing, use_tiling)
- 应用序列并行 (如果启用)

#### 2.5 OvisImagePipeline 初始化 (`pipeline_ovis_image.py`)

```python
# 关键日志点：
----my_debug---- [OvisImagePipeline.__init__] od_config.model=<模型路径>
----my_debug---- [OvisImagePipeline.__init__] _execution_device=cuda:0
----my_debug---- [OvisImagePipeline.__init__] Loading scheduler from <path>/scheduler
----my_debug---- [OvisImagePipeline.__init__] Loading text_encoder from <path>/text_encoder
----my_debug---- [OvisImagePipeline.__init__] Loading VAE from <path>/vae
----my_debug---- [OvisImagePipeline.__init__] Loading tokenizer from <path>/tokenizer
----my_debug---- [OvisImagePipeline.__init__] Initializing OvisImageTransformer2DModel
```

加载的组件及其来源：

| 组件 | 类型 | 加载来源 | 配置文件 |
|------|------|----------|----------|
| scheduler | FlowMatchEulerDiscreteScheduler | `<model>/scheduler/` | scheduler_config.json |
| text_encoder | Qwen3Model | `<model>/text_encoder/` | config.json |
| vae | AutoencoderKL | `<model>/vae/` | config.json |
| tokenizer | Qwen2TokenizerFast | `<model>/tokenizer/` | tokenizer_config.json |
| transformer | OvisImageTransformer2DModel | 自定义实现 | od_config.tf_model_config |

关键配置示例：

```python
# VAE 配置 (vae/config.json)
{
    "block_out_channels": [128, 256, 512, 512],
    "latent_channels": 16,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159
}
# vae_scale_factor = 2^(4-1) = 8

# Text Encoder 配置
{
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "vocab_size": 151936
}
```

#### 2.6 OvisImageTransformer2DModel 初始化 (`ovis_image_transformer.py`)

```python
# 关键日志点：
----my_debug---- [OvisImageTransformer2DModel.__init__] in_channels=64, out_channels=64, inner_dim=3072
----my_debug---- [OvisImageTransformer2DModel.__init__] Creating 19 OvisImageTransformerBlock layers
----my_debug---- [OvisImageTransformer2DModel.__init__] Creating 38 OvisImageSingleTransformerBlock layers
----my_debug---- [OvisImageTransformer2DModel.__init__] Total parameters: 2,398,464,000 (2.40B)
```

架构组成：
- `x_embedder`: Linear(in_channels=64, out_features=3072)
- `context_embedder_norm`: RMSNorm(2048)
- `context_embedder`: Linear(2048, 3072)
- `transformer_blocks`: 19 个 OvisImageTransformerBlock (双流 DiT)
- `single_transformer_blocks`: 38 个 OvisImageSingleTransformerBlock (单流 DiT)
- `norm_out`: AdaLayerNormContinuous
- `proj_out`: Linear(3072, 64)

#### 2.7 权重加载 (`ovis_image_transformer.py`)

```python
# 关键日志点：
----my_debug---- [OvisImageTransformer2DModel.load_weights] Total weights processed: 500+
----my_debug---- [OvisImageTransformer2DModel.load_weights] Loaded parameters: 500+
```

权重来源：
- 路径: `<model>/transformer/`
- 格式: safetensors
- 映射: QKV 权重合并 (to_q, to_k, to_v → to_qkv)

---

## 二、前向传播流程

### 1. 调用链

```
OvisImagePipeline.forward(req)
    ├── _get_messages()                    # 构建 prompt 消息
    ├── encode_prompt()                    # 编码文本 prompt
    │   └── _get_ovis_prompt_embeds()
    │       ├── tokenizer()                # 分词
    │       └── text_encoder()             # 文本编码
    ├── prepare_latents()                  # 准备噪声 latents
    │   └── randn_tensor()                 # 生成随机噪声
    │   └── _pack_latents()                # 打包 latents
    ├── prepare_timesteps()                # 准备时间步
    │   └── retrieve_timesteps()           # 获取时间步序列
    └── diffuse()                          # 扩散循环
        └── predict_noise_maybe_with_cfg() # 预测噪声 (带 CFG)
        │   └── OvisImageTransformer2DModel.forward()
        │       ├── x_embedder()           # 图像嵌入
        │       ├── timestep_embedder()    # 时间步嵌入
        │       ├── context_embedder()     # 文本嵌入
        │       ├── pos_embed()            # 位置编码 (RoPE)
        │       ├── transformer_blocks[]   # 双流 transformer 块
        │       ├── single_transformer_blocks[] # 单流 transformer 块
        │       ├── norm_out()             # 输出归一化
        │       └── proj_out()             # 输出投影
        └── scheduler_step_maybe_with_cfg() # 调度器步进
    └── vae.decode()                       # VAE 解码
```

### 2. 详细步骤

#### 2.1 Prompt 处理 (`pipeline_ovis_image.py`)

```python
# 关键日志点：
----my_debug---- [_get_messages] Input prompt: "a cat sitting on a chair"
----my_debug---- [_get_messages] Generated 1 messages
```

系统提示词：
```
Describe the image by detailing the color, quantity, text, shape, size, texture, spatial
relationships of the objects and background: 
```

#### 2.2 文本编码 (`pipeline_ovis_image.py`)

```python
# 关键日志点：
----my_debug---- [_get_ovis_prompt_embeds] Tokenizing with max_length=284
----my_debug---- [_get_ovis_prompt_embeds] input_ids shape=torch.Size([1, 284])
----my_debug---- [_get_ovis_prompt_embeds] attention_mask shape=torch.Size([1, 284])
----my_debug---- [_get_ovis_prompt_embeds] last_hidden_state shape=torch.Size([1, 284, 2048])
----my_debug---- [_get_ovis_prompt_embeds] After slicing from user_prompt_begin_id=28: shape=torch.Size([1, 256, 2048])
```

处理流程：
1. 应用 chat template
2. Tokenize (max_length=284, padding="max_length")
3. Text Encoder forward pass
4. 应用 attention mask
5. 切片去掉系统提示部分 (从 user_prompt_begin_id=28 开始)

#### 2.3 Latent 准备 (`pipeline_ovis_image.py`)

```python
# 关键日志点：
----my_debug---- [prepare_latents] Input: height=1024, width=1024
----my_debug---- [prepare_latents] After VAE scaling: height=64, width=64
----my_debug---- [prepare_latents] Latent shape before packing: (1, 16, 64, 64)
----my_debug---- [prepare_latents] Random noise stats: min=-3.5, max=3.5, mean=0.0, std=1.0
----my_debug---- [prepare_latents] After packing: shape=torch.Size([1, 1024, 64])
```

VAE 压缩计算：
- 输入: (H, W) = (1024, 1024)
- VAE scale factor: 8
- Latent: (H/8, W/8) = (128, 128)
- Packing 后: (H/16, W/16) = (64, 64)
- 最终形状: (batch, seq_len, channels) = (1, 64*64=4096, 16*4=64)

#### 2.4 时间步准备 (`pipeline_ovis_image.py`)

```python
# 关键日志点：
----my_debug---- [prepare_timesteps] num_inference_steps=50
----my_debug---- [prepare_timesteps] calculated mu=0.87
----my_debug---- [prepare_timesteps] timesteps values (first 10): [1000.0, 980.0, ...]
```

时间步计算：
- 使用 FlowMatchEulerDiscreteScheduler
- 应用 shift 计算 mu 值
- 生成从 1.0 到 0 的 sigma 序列

#### 2.5 Transformer Forward (`ovis_image_transformer.py`)

```python
# 关键日志点：
----my_debug---- [OvisImageTransformer2DModel.forward] hidden_states shape=torch.Size([1, 4096, 64])
----my_debug---- [OvisImageTransformer2DModel.forward] encoder_hidden_states shape=torch.Size([1, 256, 3072])
----my_debug---- [OvisImageTransformer2DModel.forward] timestep=0.5
----my_debug---- [OvisImageTransformer2DModel.forward] image_rotary_emb: cos shape=torch.Size([4352, 48])
```

处理流程：
1. **x_embedder**: (batch, seq_len, 64) → (batch, seq_len, 3072)
2. **timestep embedding**: timestep → (batch, 3072)
3. **context_embedder**: (batch, 256, 2048) → (batch, 256, 3072)
4. **position embedding**: 生成 RoPE (cos, sin)
5. **dual-stream blocks** (19 层): 
   - OvisImageTransformerBlock 包含 self-attention 和 cross-attention
6. **single-stream blocks** (38 层):
   - OvisImageSingleTransformerBlock 合并处理
7. **norm_out + proj_out**: (batch, seq_len, 3072) → (batch, seq_len, 64)

#### 2.6 Attention 计算 (`ovis_image_transformer.py`)

```python
# 关键日志点：
----my_debug---- [OvisImageAttention.forward] hidden_states shape=torch.Size([1, 4096, 3072])
----my_debug---- [OvisImageAttention.forward] qkv shape after to_qkv=torch.Size([1, 4096, 9216])
----my_debug---- [OvisImageAttention.forward] query shape after unflatten=torch.Size([1, 4096, 24, 128])
----my_debug---- [OvisImageAttention.forward] Running attention: query=[1, 4352, 24, 128]
```

Attention 参数：
- heads: 24
- head_dim: 128
- inner_dim: 3072
- RoPE 应用于 query 和 key

#### 2.7 VAE 解码 (`pipeline_ovis_image.py`)

```python
# 关键日志点：
----my_debug---- [forward] Unpacking latents from shape torch.Size([1, 4096, 64])
----my_debug---- [forward] Unpacked latents shape=torch.Size([1, 16, 128, 128])
----my_debug---- [forward] VAE config: scaling_factor=0.3611, shift_factor=0.1159
----my_debug---- [forward] VAE decode output shape=torch.Size([1, 3, 1024, 1024])
```

解码流程：
1. Unpack latents: (1, 4096, 64) → (1, 16, 128, 128)
2. 反缩放: latents / scaling_factor + shift_factor
3. VAE decode: (1, 16, 128, 128) → (1, 3, 1024, 1024)

---

## 三、关键数据结构

### 1. 配置结构

```python
OmniDiffusionConfig:
    model: str                    # 模型路径或 HuggingFace ID
    model_class_name: str         # 模型架构名称
    dtype: torch.dtype           # 数据类型 (如 torch.bfloat16)
    tf_model_config: dict        # Transformer 配置
        num_layers: int          # 双流 transformer 层数
        num_single_layers: int   # 单流 transformer 层数
        attention_head_dim: int  # 注意力头维度
        num_attention_heads: int # 注意力头数量
    parallel_config: dict        # 并行配置
        tensor_parallel_size: int
        sequence_parallel_size: int
        use_hsdp: bool
    vae_use_tiling: bool
    vae_use_slicing: bool
```

### 2. 张量形状总结

| 阶段 | 张量名称 | 形状 | 说明 |
|------|----------|------|------|
| 文本编码 | input_ids | (batch, 284) | tokenized 输入 |
| 文本编码 | prompt_embeds | (batch, 256, 2048) | 文本嵌入 |
| 噪声生成 | latents | (batch, 16, 128, 128) | 随机噪声 |
| 打包后 | latents_packed | (batch, 4096, 64) | 打包的 latents |
| Transformer 输入 | hidden_states | (batch, 4096, 64) | 图像 tokens |
| Transformer 中间 | hidden_states | (batch, 4096, 3072) | 嵌入后 |
| Transformer 输出 | output | (batch, 4096, 64) | 噪声预测 |
| VAE 输入 | latents | (batch, 16, 128, 128) | 解包后 |
| VAE 输出 | image | (batch, 3, 1024, 1024) | 最终图像 |

---

## 四、日志使用说明

### 1. 启用日志

所有日志使用 vLLM 的 logger，日志级别为 INFO。日志前缀为 `----my_debug----`。

### 2. 过滤日志

使用 grep 过滤日志：
```bash
python your_script.py 2>&1 | grep "----my_debug----"
```

### 3. 关键日志标签

- `[DiffusionWorker.__init__]`: Worker 初始化
- `[init_device]`: 设备初始化
- `[load_model]`: 模型加载
- `[initialize_model]`: 模型实例化
- `[OvisImagePipeline.__init__]`: Pipeline 初始化
- `[OvisImageTransformer2DModel.__init__]`: Transformer 初始化
- `[OvisImageTransformer2DModel.load_weights]`: 权重加载
- `[forward]`: 前向传播
- `[_get_ovis_prompt_embeds]`: 文本编码
- `[prepare_latents]`: 噪声准备
- `[prepare_timesteps]`: 时间步准备
- `[diffuse]`: 扩散循环
- `[OvisImageTransformer2DModel.forward]`: Transformer 前向
- `[OvisImageAttention.forward]`: Attention 计算

---

## 五、修改的文件列表

1. `vllm_omni/diffusion/models/ovis_image/pipeline_ovis_image.py`
2. `vllm_omni/diffusion/models/ovis_image/ovis_image_transformer.py`
3. `vllm_omni/diffusion/model_loader/diffusers_loader.py`
4. `vllm_omni/diffusion/worker/diffusion_model_runner.py`
5. `vllm_omni/diffusion/worker/diffusion_worker.py`
6. `vllm_omni/diffusion/registry.py`
