# Cache Backend 与 Offload Backend 详解

## 一、总览

DiffusionModelRunner 在模型加载后会初始化两个关键的后端：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DiffusionModelRunner.load_model()                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
        ▼                                                       ▼
┌───────────────────────┐                           ┌───────────────────────┐
│   offload_backend     │                           │    cache_backend      │
│                       │                           │                       │
│ ┌───────────────────┐ │                           │ ┌───────────────────┐ │
│ │ ModelLevelOffload │ │                           │ │   TeaCacheBackend │ │
│ │ (模型级卸载)       │ │                           │ │   (TeaCache)      │ │
│ └───────────────────┘ │                           │ └───────────────────┘ │
│ ┌───────────────────┐ │                           │ ┌───────────────────┐ │
│ │ LayerWiseOffload  │ │                           │ │ CacheDiTBackend   │ │
│ │ (层级卸载)        │ │                           │ │ (Cache-DiT)       │ │
│ └───────────────────┘ │                           │ └───────────────────┘ │
└───────────────────────┘                           └───────────────────────┘
        │                                                       │
        ▼                                                       ▼
┌───────────────────────┐                           ┌───────────────────────┐
│     内存优化           │                           │     推理加速          │
│ - 减少 GPU 显存占用   │                           │ - 减少计算量          │
│ - 支持大模型运行      │                           │ - 提高吞吐量          │
└───────────────────────┘                           └───────────────────────┘
```

---

## 二、Cache Backend（缓存后端）

### 2.1 作用与目的

**核心目的**：通过缓存中间计算结果，跳过不必要的计算，加速扩散模型推理。

```
传统推理：
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 0: Transformer 计算                                               │
│  Step 1: Transformer 计算                                               │
│  Step 2: Transformer 计算                                               │
│  ...                                                                    │
│  Step 49: Transformer 计算                                              │
│                                                                         │
│  总计算量: 50 次完整的 Transformer 前向传播                              │
└─────────────────────────────────────────────────────────────────────────┘

缓存加速后：
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 0: Transformer 计算 (Warmup)                                      │
│  Step 1: 复用缓存 (跳过计算)                                            │
│  Step 2: Transformer 计算 (缓存过期)                                    │
│  Step 3: 复用缓存 (跳过计算)                                            │
│  ...                                                                    │
│  Step 49: 复用缓存                                                      │
│                                                                         │
│  总计算量: 约 20 次完整的 Transformer 前向传播 (节省 60%)                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 两种缓存后端对比

| 特性 | TeaCache | Cache-DiT |
|------|----------|-----------|
| **实现方式** | Hook 拦截前向传播 | cache-dit 库集成 |
| **缓存粒度** | 整体 Transformer | Block 级别 |
| **加速机制** | Timestep Embedding 相似性 | DBCache, SCM, TaylorSeer |
| **配置复杂度** | 简单 (rel_l1_thresh) | 复杂 (多个参数) |
| **适用场景** | 通用加速 | 细粒度控制、高级特性 |
| **官方仓库** | 无独立仓库 | https://github.com/vipshop/cache-dit |

### 2.3 代码位置

```
vllm_omni/diffusion/cache/
├── __init__.py                    # 导出接口
├── base.py                        # CacheBackend 基类
├── selector.py                    # 后端选择器
├── cache_dit_backend.py           # Cache-DiT 后端实现
└── teacache/
    ├── __init__.py
    ├── backend.py                 # TeaCache 后端实现
    ├── config.py                  # TeaCache 配置
    ├── hook.py                    # TeaCache Hook 实现
    ├── state.py                   # TeaCache 状态管理
    └── extractors.py              # 模型特定的特征提取器
```

---

## 三、TeaCache 详解

### 3.1 原理

**TeaCache (Timestep Embedding Aware Cache)** 基于 Diffusion Transformer 的特性：
- 相邻时间步的输入特征相似
- 可以跳过相似时间步的计算，复用残差

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TeaCache 核心算法                                │
└─────────────────────────────────────────────────────────────────────────┘

Step t:
  输入: hidden_states_t, timestep_t

  提取 modulated_input = first_block(hidden_states_t, timestep_t)

  计算 L1 距离:
    rel_l1 = |modulated_input_t - modulated_input_{t-1}|
             / |modulated_input_{t-1}|

  多项式缩放 (模型特定系数):
    rescaled = poly(rel_l1)

  累积距离:
    accumulated += rescaled

  决策:
    if accumulated < threshold:
        使用缓存: hidden_states = hidden_states + previous_residual
    else:
        完整计算: residual = transformer(hidden_states) - hidden_states
        更新缓存
```

### 3.2 生命周期

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TeaCache 生命周期                                 │
└─────────────────────────────────────────────────────────────────────────┘

1. 创建阶段
   ┌─────────────────────────────────────────────────────────────────────┐
   │ TeaCacheBackend(config)                                             │
   │   └── config.rel_l1_thresh = 0.2                                   │
   └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. 启用阶段 (load_model 时调用)
   ┌─────────────────────────────────────────────────────────────────────┐
   │ backend.enable(pipeline)                                            │
   │   ├── 创建 TeaCacheConfig                                          │
   │   ├── 获取 transformer_type = pipeline.transformer.__class__.__name│
   │   ├── 获取模型特定的 coefficients                                  │
   │   └── apply_teacache_hook(transformer, config)                     │
   │         └── 注册 TeaCacheHook 到 HookRegistry                      │
   └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. 推理阶段 (每次生成)
   ┌─────────────────────────────────────────────────────────────────────┐
   │ backend.refresh(pipeline, num_inference_steps)                      │
   │   └── 重置所有缓存状态                                              │
   │                                                                      │
   │ 每个 step:                                                           │
   │   TeaCacheHook.new_forward()                                        │
   │     ├── 提取 modulated_input                                        │
   │     ├── 计算 L1 距离                                                │
   │     ├── 决策: 使用缓存 or 完整计算                                  │
   │     └── 更新状态                                                    │
   └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
4. 销毁阶段
   ┌─────────────────────────────────────────────────────────────────────┐
   │ pipeline 销毁时自动清理                                              │
   └─────────────────────────────────────────────────────────────────────┘
```

### 3.3 核心调用逻辑

```python
# 文件: vllm_omni/diffusion/cache/teacache/hook.py

class TeaCacheHook(ModelHook):
    def new_forward(self, module, *args, **kwargs):
        # 1. 获取模型特定的提取器
        ctx = self.extractor_fn(module, *args, **kwargs)

        # 2. 设置 CFG 分支上下文
        context_name = f"teacache_{cache_branch}"
        state = self.state_manager.get_state()

        # 3. 决策是否使用缓存
        should_compute = self._should_compute_full_transformer(
            state, ctx.modulated_input
        )

        if not should_compute and state.previous_residual is not None:
            # 快速路径: 复用缓存
            ctx.hidden_states = ctx.hidden_states + state.previous_residual
        else:
            # 慢速路径: 完整计算
            outputs = ctx.run_transformer_blocks()
            state.previous_residual = ctx.hidden_states - ori_hidden_states

        return ctx.postprocess(output)

    def _should_compute_full_transformer(self, state, modulated_inp):
        # TeaCache 核心算法
        if state.cnt == 0:
            return True  # 第一步必须计算

        # 计算 L1 距离
        rel_distance = (
            (modulated_inp - state.previous_modulated_input).abs().mean()
            / state.previous_modulated_input.abs().mean()
        ).cpu().item()

        # 多项式缩放
        rescaled_distance = self.rescale_func(rel_distance)
        state.accumulated_rel_l1_distance += abs(rescaled_distance)

        # 与阈值比较
        return state.accumulated_rel_l1_distance >= self.config.rel_l1_thresh
```

### 3.4 配置参数

```python
# 文件: vllm_omni/diffusion/cache/teacache/config.py

@dataclass
class TeaCacheConfig:
    rel_l1_thresh: float = 0.2      # L1 距离阈值
    coefficients: list[float] | None = None  # 多项式系数 (None 则自动选择)
    transformer_type: str = "QwenImageTransformer2DModel"  # 模型类型
```

**rel_l1_thresh 参数说明**：
- `0.1-0.15`: 保守，质量高，加速约 1.2-1.3x
- `0.2`: 平衡，质量损失小，加速约 1.5x (推荐)
- `0.3-0.4`: 激进，轻微质量损失，加速约 1.6-1.8x
- `0.5+`: 非常激进，质量损失明显，加速 2x+

### 3.5 模型特定系数

```python
# 文件: vllm_omni/diffusion/cache/teacache/config.py

_MODEL_COEFFICIENTS = {
    "FluxTransformer2DModel": [4.9865e02, -2.8378e02, 5.5855e01, -3.8202e00, 2.6423e-01],
    "QwenImageTransformer2DModel": [-4.5e02, 2.8e02, -4.5e01, 3.2e00, -2e-02],
    "Bagel": [1.333e06, -1.686e05, 7.95e03, -1.637e02, 1.263e00],
    "StableAudioDiTModel": [121.77, -153.74, 68.05, -12.28, 1.07],
    # ... 更多模型
}
```

---

## 四、Cache-DiT 详解

### 4.1 原理

**Cache-DiT** 是一个功能更强大的缓存库，提供多种加速技术：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Cache-DiT 核心技术                                │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│     DBCache       │   │       SCM         │   │   TaylorSeer      │
│  (Diffusion Block │   │ (Step Computation │   │   (Taylor Series  │
│      Cache)       │   │      Masking)     │   │   Approximation)  │
└───────────────────┘   └───────────────────┘   └───────────────────┘
        │                       │                       │
        │ 缓存 block 级别的    │ 智能选择哪些步     │ 使用泰勒级数
        │ 中间结果             │ 需要计算           │ 预测特征
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                                    │
                                    ▼
                          组合使用获得最佳加速效果
```

### 4.2 生命周期

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Cache-DiT 生命周期                                 │
└─────────────────────────────────────────────────────────────────────────┘

1. 创建阶段
   ┌─────────────────────────────────────────────────────────────────────┐
   │ CacheDiTBackend(cache_config)                                       │
   │   ├── config.Fn_compute_blocks = 1     # 前向计算块数              │
   │   ├── config.Bn_compute_blocks = 0     # 后向计算块数              │
   │   ├── config.max_warmup_steps = 4      # 预热步数                  │
   │   ├── config.residual_diff_threshold = 0.24  # 残差阈值            │
   │   └── config.enable_taylorseer = False  # 是否启用 TaylorSeer      │
   └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. 启用阶段
   ┌─────────────────────────────────────────────────────────────────────┐
   │ backend.enable(pipeline)                                            │
   │   ├── 检查是否在 CUSTOM_DIT_ENABLERS 中                           │
   │   │   └── 是: 调用模型特定的 enable 函数                           │
   │   │   └── 否: 调用通用 enable_cache_for_dit()                      │
   │   │                                                                  │
   │   ├── 创建 BlockAdapter (描述 Transformer 结构)                    │
   │   │   ├── transformer: 目标 Transformer                            │
   │   │   ├── blocks: 要缓存的 block 列表                              │
   │   │   ├── forward_pattern: 前向传播模式                            │
   │   │   └── params_modifiers: 参数修改器                             │
   │   │                                                                  │
   │   └── cache_dit.enable_cache(adapter, cache_config)                │
   │         └── 注入缓存逻辑到 Transformer                             │
   └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. 推理阶段
   ┌─────────────────────────────────────────────────────────────────────┐
   │ backend.refresh(pipeline, num_inference_steps)                      │
   │   ├── 检查 num_inference_steps 是否变化                            │
   │   └── 调用 _refresh_func(pipeline, num_inference_steps)            │
   │         └── cache_dit.refresh_context(transformer, ...)            │
   │                                                                      │
   │ 每个 step:                                                           │
   │   CachedBlocks.forward()                                            │
   │     ├── 检查是否可以复用缓存                                        │
   │     ├── 是: 应用缓存的残差                                          │
   │     └── 否: 计算完整 block 并缓存结果                               │
   └─────────────────────────────────────────────────────────────────────┘
```

### 4.3 核心调用逻辑

```python
# 文件: vllm_omni/diffusion/cache/cache_dit_backend.py

class CacheDiTBackend(CacheBackend):
    def enable(self, pipeline):
        pipeline_name = pipeline.__class__.__name__

        # 检查是否有自定义 enabler
        if pipeline_name in CUSTOM_DIT_ENABLERS:
            self._refresh_func = CUSTOM_DIT_ENABLERS[pipeline_name](pipeline, self.config)
        else:
            self._refresh_func = enable_cache_for_dit(pipeline, self.config)

        self.enabled = True

    def refresh(self, pipeline, num_inference_steps, verbose=True):
        if num_inference_steps != self._last_num_inference_steps:
            self._refresh_func(pipeline, num_inference_steps, verbose)
            self._last_num_inference_steps = num_inference_steps
```

### 4.4 模型特定配置

不同模型需要不同的 BlockAdapter 配置：

```python
# Flux 模型 (双流架构)
def enable_cache_for_flux(pipeline, cache_config):
    cache_dit.enable_cache(
        BlockAdapter(
            transformer=pipeline.transformer,
            blocks=[
                pipeline.transformer.transformer_blocks,      # 主 blocks
                pipeline.transformer.single_transformer_blocks, # 单流 blocks
            ],
            forward_pattern=[ForwardPattern.Pattern_1, ForwardPattern.Pattern_1],
            params_modifiers=[modifier],
        ),
        cache_config=db_cache_config,
    )

# Wan2.2 模型 (双 Transformer 架构)
def enable_cache_for_wan22(pipeline, cache_config):
    if pipeline.transformer_2 is None:
        # 单 Transformer 模式
        cache_dit.enable_cache(
            BlockAdapter(
                transformer=pipeline.transformer,
                blocks=[pipeline.transformer.blocks],
                forward_pattern=[ForwardPattern.Pattern_2],
            ),
            ...
        )
    else:
        # 双 Transformer 模式 (高噪声 + 低噪声)
        cache_dit.enable_cache(
            BlockAdapter(
                transformer=[pipeline.transformer, pipeline.transformer_2],
                blocks=[pipeline.transformer.blocks, pipeline.transformer_2.blocks],
                forward_pattern=[ForwardPattern.Pattern_2, ForwardPattern.Pattern_2],
                params_modifiers=[
                    ParamsModifier(cache_config=high_noise_config),
                    ParamsModifier(cache_config=low_noise_config),
                ],
            ),
            ...
        )
```

### 4.5 ForwardPattern 说明

Cache-DiT 定义了多种前向传播模式：

| Pattern | 输入 | 输出 | 适用模型 |
|---------|------|------|----------|
| Pattern_0 | (hidden, encoder_hidden) | (hidden, encoder_hidden) | Bagel, GlmImage |
| Pattern_1 | (hidden, encoder_hidden) | hidden | Flux, SD3 |
| Pattern_2 | hidden | hidden | Wan2.2 |
| Pattern_3 | hidden | hidden (cross-attn 内部) | StableAudio |
| Pattern_4 | hidden | (hidden,) | HunyuanImage3 |

### 4.6 配置参数详解

```python
# 文件: vllm_omni/diffusion/data.py

@dataclass
class DiffusionCacheConfig:
    # TeaCache 参数
    rel_l1_thresh: float = 0.2           # L1 距离阈值
    coefficients: list[float] | None = None  # 多项式系数

    # Cache-DiT 参数
    Fn_compute_blocks: int = 1           # 前向计算块数
    Bn_compute_blocks: int = 0           # 后向计算块数
    max_warmup_steps: int = 4            # 预热步数
    max_cached_steps: int = -1           # 最大缓存步数 (-1 = 无限)
    residual_diff_threshold: float = 0.24  # 残差阈值
    max_continuous_cached_steps: int = 3   # 最大连续缓存步数

    # TaylorSeer 配置
    enable_taylorseer: bool = False
    taylorseer_order: int = 1

    # SCM (Step Computation Masking)
    scm_steps_mask_policy: str | None = None
    scm_steps_policy: str = "dynamic"
```

---

## 五、Offload Backend（卸载后端）

### 5.1 作用与目的

**核心目的**：通过在 CPU 和 GPU 之间动态移动模型权重，减少 GPU 显存占用。

```
场景：GPU 显存不足，无法同时容纳所有模型组件

┌─────────────────────────────────────────────────────────────────────────┐
│                     无 Offloading (OOM)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  GPU: [Transformer (20GB) | TextEncoder (5GB) | VAE (3GB)]              │
│                                                                         │
│  总需求: 28GB > GPU 显存 24GB → OOM!                                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   Model-Level Offloading                                │
├─────────────────────────────────────────────────────────────────────────┤
│  推理阶段 1 (编码):                                                     │
│    GPU: [TextEncoder (5GB)]  ← Transformer 在 CPU                      │
│                                                                         │
│  推理阶段 2 (去噪):                                                     │
│    GPU: [Transformer (20GB)] ← TextEncoder 在 CPU                      │
│                                                                         │
│  推理阶段 3 (解码):                                                     │
│    GPU: [VAE (3GB)]  ← Transformer 在 CPU                              │
│                                                                         │
│  峰值显存: 20GB < 24GB ✓                                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   Layer-Wise Offloading                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  每个时刻只有 1-2 个 block 在 GPU:                                      │
│                                                                         │
│  GPU: [Block_i (0.5GB) | Block_{i+1} (prefetch, 0.5GB)]               │
│                                                                         │
│  峰值显存: ~1GB (仅为 Transformer 的 5%)                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 两种 Offload 策略

| 特性 | Model-Level Offloading | Layer-Wise Offloading |
|------|------------------------|----------------------|
| **粒度** | 模型级别 (Transformer vs Encoder) | Block 级别 |
| **显存节省** | 约 50% | 约 90%+ |
| **性能影响** | 中等 (阶段切换开销) | 较大 (每步都有传输) |
| **适用场景** | 中等显存压力 | 极端显存压力 |
| **实现复杂度** | 简单 | 复杂 (异步预取) |

### 5.3 代码位置

```
vllm_omni/diffusion/offloader/
├── __init__.py                    # 导出接口
├── base.py                        # OffloadBackend 基类
├── sequential_backend.py          # Model-Level Offloading
├── layerwise_backend.py           # Layer-Wise Offloading
└── module_collector.py            # 模块发现与收集
```

---

## 六、Model-Level Offloading 详解

### 6.1 工作原理

**互斥访问模式**：DiT 和 Encoder 互斥使用 GPU。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Model-Level Offloading 流程                          │
└─────────────────────────────────────────────────────────────────────────┘

时间轴 ─────────────────────────────────────────────────────────────────────►

Step 1: 文本编码
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Encoder.forward() 触发:                                                │
│    ├── pre_forward hook:                                               │
│    │     ├── Transformer._to_cpu()  # 卸载 Transformer 到 CPU         │
│    │     └── Encoder._to_gpu()       # 加载 Encoder 到 GPU            │
│    │                                                                    │
│    └── 执行 Encoder 计算                                                │
│                                                                         │
│  GPU: [Encoder]  CPU: [Transformer, VAE]                               │
└─────────────────────────────────────────────────────────────────────────┘

Step 2-50: 去噪循环
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Transformer.forward() 触发:                                            │
│    ├── pre_forward hook:                                               │
│    │     ├── Encoder._to_cpu()      # 卸载 Encoder 到 CPU             │
│    │     └── Transformer._to_gpu()  # 加载 Transformer 到 GPU         │
│    │                                                                    │
│    └── 执行 Transformer 计算                                            │
│                                                                         │
│  GPU: [Transformer]  CPU: [Encoder, VAE]                               │
└─────────────────────────────────────────────────────────────────────────┘

Step 51: 图像解码
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  VAE.forward() 触发:                                                    │
│    ├── pre_forward hook:                                               │
│    │     ├── Transformer._to_cpu()  # 卸载 Transformer 到 CPU         │
│    │     └── VAE._to_gpu()          # 加载 VAE 到 GPU                  │
│    │                                                                    │
│    └── 执行 VAE 解码                                                    │
│                                                                         │
│  GPU: [VAE]  CPU: [Transformer, Encoder]                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 核心实现

```python
# 文件: vllm_omni/diffusion/offloader/sequential_backend.py

class SequentialOffloadHook(ModelHook):
    """互斥卸载 Hook"""

    def __init__(self, offload_targets, device, pin_memory=True, use_hsdp=False):
        self.offload_targets = offload_targets  # 需要卸载的目标模块
        self.device = device
        self.pin_memory = pin_memory

    def pre_forward(self, module, *args, **kwargs):
        # 1. 卸载目标模块到 CPU
        for target in self.offload_targets:
            self._to_cpu(target)

        # 2. 加载当前模块到 GPU
        self._to_gpu(module)

        # 3. 同步 GPU
        current_omni_platform.synchronize()

        return args, kwargs

    def _to_cpu(self, module):
        # 移动参数和缓冲区到 CPU
        self._move_params(module, torch.device("cpu"), pin_memory=self.pin_memory)
        current_omni_platform.empty_cache()

    def _to_gpu(self, module):
        # 移动参数和缓冲区到 GPU
        self._move_params(module, self.device, non_blocking=False)


def apply_sequential_offload(dit_modules, encoder_modules, device, ...):
    """应用互斥卸载 Hook"""

    # DiT 模块运行时卸载 Encoder
    for dit_mod in dit_modules:
        registry = HookRegistry.get_or_create(dit_mod)
        hook = SequentialOffloadHook(
            offload_targets=encoder_modules,
            device=device,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)

    # Encoder 运行时卸载 DiT
    for enc in encoder_modules:
        registry = HookRegistry.get_or_create(enc)
        hook = SequentialOffloadHook(
            offload_targets=dit_modules,
            device=device,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
```

### 6.3 生命周期

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Model-Level Offloading 生命周期                         │
└─────────────────────────────────────────────────────────────────────────┘

1. 创建后端
   ┌─────────────────────────────────────────────────────────────────────┐
   │ backend = ModelLevelOffloadBackend(config, device)                  │
   └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. 启用 (load_model 时)
   ┌─────────────────────────────────────────────────────────────────────┐
   │ backend.enable(pipeline)                                            │
   │   ├── ModuleDiscovery.discover(pipeline)                            │
   │   │     ├── 发现 DiT 模块: [transformer, transformer_2]            │
   │   │     ├── 发现 Encoder 模块: [text_encoder, text_encoder_2]      │
   │   │     └── 发现 VAE 模块: [vae]                                    │
   │   │                                                                  │
   │   ├── 移动 Encoder 到 GPU                                           │
   │   ├── 移动 VAE 到 GPU                                               │
   │   │                                                                  │
   │   └── apply_sequential_offload(dits, encoders, device)              │
   │         ├── 注册 Hook 到 DiT 模块                                   │
   │         └── 注册 Hook 到 Encoder 模块                               │
   └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. 推理阶段
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 每次 forward:                                                       │
   │   SequentialOffloadHook.pre_forward()                               │
   │     ├── 卸载目标模块                                                │
   │     └── 加载当前模块                                                │
   └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
4. 禁用
   ┌─────────────────────────────────────────────────────────────────────┐
   │ backend.disable()                                                   │
   │   └── remove_sequential_offload(modules)                            │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## 七、Layer-Wise Offloading 详解

### 7.1 工作原理

**滑动窗口模式**：每个时刻只有少量 Block 在 GPU，通过异步预取实现计算与传输重叠。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Layer-Wise Offloading 流程                          │
└─────────────────────────────────────────────────────────────────────────┘

假设 Transformer 有 40 个 Block:

传统方式 (全部在 GPU):
┌─────────────────────────────────────────────────────────────────────────┐
│  GPU: [Block_0 | Block_1 | ... | Block_39]                             │
│  显存占用: 40 * 0.5GB = 20GB                                            │
└─────────────────────────────────────────────────────────────────────────┘

Layer-Wise Offloading:
┌─────────────────────────────────────────────────────────────────────────┐
│  时间 t: 计算 Block_i                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  GPU: [Block_i (当前计算) | Block_{i+1} (预取中)]                │  │
│  │                                                                   │  │
│  │  计算流:                                                          │  │
│  │    ├── Block_i.forward()                                         │  │
│  │    └── post_forward: Block_i.offload_layer()                     │  │
│  │                                                                   │  │
│  │  拷贝流 (异步):                                                   │  │
│  │    └── Block_{i+1}.prefetch_layer() (CPU → GPU)                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  时间 t+1: 计算 Block_{i+1}                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  GPU: [Block_{i+1} (已预取) | Block_{i+2} (预取中)]              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  显存占用: 2 * 0.5GB = 1GB (仅为传统的 5%)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 核心实现

```python
# 文件: vllm_omni/diffusion/offloader/layerwise_backend.py

class LayerwiseOffloadHook(ModelHook):
    """层级卸载 Hook"""

    def __init__(self, next_block, device, stream, pin_memory=True):
        self.next_block = next_block      # 下一个要预取的 block
        self.device = device
        self.copy_stream = stream         # 异步拷贝流

        # 预分配 CPU 扁平化权重存储
        self.dtype_cpu_flattened_weights = {}
        self.dtype_metadata = {}

    def initialize_hook(self, module):
        # 初始化: 将 next_block 的权重扁平化存储到 CPU
        self.dtype_cpu_flattened_weights, self.dtype_metadata = \
            self._to_cpu(self.next_block_parameters, self.next_block_buffers, ...)
        return module

    def pre_forward(self, module, *args, **kwargs):
        # 预取下一层权重 (异步)
        self.prefetch_layer(non_blocking=True)
        return args, kwargs

    def post_forward(self, module, output):
        # 释放当前层 GPU 显存
        self.offload_layer()
        return output

    @torch.compiler.disable
    def prefetch_layer(self, non_blocking=True):
        """异步预取下一层权重: CPU → GPU"""
        self.copy_stream.wait_stream(current_omni_platform.current_stream())

        gpu_weights = {}
        with current_omni_platform.stream(self.copy_stream):
            # 在拷贝流上异步传输
            for dtype, cpu_weight in self.dtype_cpu_flattened_weights.items():
                gpu_weight = torch.empty(cpu_weight.shape, dtype=dtype, device=self.device)
                gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
                gpu_weights[dtype] = gpu_weight

        # 将 GPU 权重视图绑定到参数
        for metadata in self.dtype_metadata[dtype]:
            target_param = self.next_block_parameters[metadata["name"]]
            target_param.data = gpu_weight[metadata["offset"]:offset+metadata["numel"]].view(metadata["shape"])

    def offload_layer(self):
        """释放当前层 GPU 显存"""
        # 等待预取完成
        if self._prefetch_done is not None:
            current_omni_platform.current_stream().wait_event(self._prefetch_done)

        # 替换为空占位符 (释放显存)
        for param in self.block_parameters.values():
            param.data = torch.empty((0,), device=self.device, dtype=param.dtype)


class LayerWiseOffloadBackend(OffloadBackend):
    def enable(self, pipeline):
        modules = ModuleDiscovery.discover(pipeline)

        for dit_module in modules.dits:
            blocks = self.get_blocks_from_dit(dit_module)

            # 为每个 block 注册 hook
            last_block, first_block = blocks[-1], blocks[0]

            # 最后一个 block 预取第一个 block (循环)
            last_hook = apply_block_hook(last_block, first_block, ...)
            last_hook.prefetch_layer(non_blocking=False)  # 初始预取

            # 其他 block 预取下一个 block
            for i, block in enumerate(blocks[:-1]):
                next_block = blocks[i + 1]
                apply_block_hook(block, next_block, ...)
```

### 7.3 关键优化

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Layer-Wise 关键优化技术                              │
└─────────────────────────────────────────────────────────────────────────┘

1. 权重扁平化
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 传统方式:                                                           │
   │   param_1: [shape_1] → 多次小传输                                  │
   │   param_2: [shape_2]                                               │
   │   ...                                                              │
   │   param_n: [shape_n]                                               │
   │                                                                     │
   │ 扁平化后:                                                           │
   │   cpu_tensor: [all_params_flattened] → 一次大传输                  │
   │   metadata: [{name, offset, numel, shape}, ...]                    │
   └─────────────────────────────────────────────────────────────────────┘

2. Pinned Memory
   ┌─────────────────────────────────────────────────────────────────────┐
   │ pin_memory=True:                                                    │
   │   CPU 的 page-locked memory → 更快的 DMA 传输                       │
   │   传输速度: ~12 GB/s (vs ~6 GB/s without pinning)                   │
   └─────────────────────────────────────────────────────────────────────┘

3. 异步拷贝流
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 计算流:                          拷贝流:                           │
   │   Block_i.forward() ─────┐                                         │
   │                          ├── 并行执行                               │
   │                          │    Block_{i+1}.prefetch() ─────         │
   │   Block_i.offload() ─────┘                                         │
   └─────────────────────────────────────────────────────────────────────┘

4. 循环预取
   ┌─────────────────────────────────────────────────────────────────────┐
   │ Block_39 (最后) 预取 Block_0 (第一个)                               │
   │ → 下一个 request 的第一步已经有 Block_0 在 GPU                      │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## 八、后端交互与调用位置

### 8.1 调用位置总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DiffusionModelRunner.load_model()                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ├──────────────────────────────────────┐
                                    │                                      │
                                    ▼                                      ▼
                    ┌───────────────────────────────┐      ┌───────────────────────────────┐
                    │       模型加载完成            │      │       应用 Offloading         │
                    │                               │      │                               │
                    │ pipeline = model_loader       │      │ offload_backend = get_offload │
                    │   .load_model(...)            │      │   _backend(od_config)         │
                    └───────────────┬───────────────┘      └───────────────┬───────────────┘
                                    │                                      │
                                    ▼                                      ▼
                    ┌───────────────────────────────┐      ┌───────────────────────────────┐
                    │     应用 CPU Offloading       │      │    应用 torch.compile         │
                    │                               │      │                               │
                    │ if enable_cpu_offload:        │      │ if not enforce_eager:         │
                    │   offload_backend.enable()    │      │   _compile_transformer()      │
                    └───────────────┬───────────────┘      └───────────────┬───────────────┘
                                    │                                      │
                                    ▼                                      ▼
                    ┌───────────────────────────────┐      ┌───────────────────────────────┐
                    │     应用 torch.compile        │      │     设置缓存后端              │
                    │                               │      │                               │
                    │ if not enforce_eager:         │      │ cache_backend = get_cache     │
                    │   _compile_transformer()      │      │   _backend(...)               │
                    └───────────────┬───────────────┘      └───────────────┬───────────────┘
                                    │                                      │
                                    ▼                                      ▼
                    ┌───────────────────────────────┐      ┌───────────────────────────────┐
                    │     设置缓存后端              │      │    cache_backend.enable()     │
                    │                               │      │                               │
                    │ cache_backend = get_cache     │      │ if cache_backend:             │
                    │   _backend(...)               │      │   cache_backend.enable()      │
                    └───────────────────────────────┘      └───────────────────────────────┘
```

### 8.2 execute_model 中的调用

```python
# 文件: vllm_omni/diffusion/worker/diffusion_model_runner.py

def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
    with grad_context:
        # 1. 刷新缓存上下文 (每次生成开始时)
        if self.cache_backend is not None and self.cache_backend.is_enabled():
            self.cache_backend.refresh(
                self.pipeline,
                req.sampling_params.num_inference_steps
            )

        # 2. 执行前向传播
        #    - Offloading 通过 Hook 自动处理
        #    - Cache 通过 Hook 或 BlockAdapter 自动处理
        with set_forward_context(...):
            output = self.pipeline.forward(req)

        return output
```

### 8.3 后端组合使用

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        后端组合使用场景                                  │
└─────────────────────────────────────────────────────────────────────────┘

场景 1: 标准推理 (GPU 显存充足)
┌─────────────────────────────────────────────────────────────────────────┐
│ config:                                                                 │
│   cache_backend: "tea_cache"                                            │
│   enable_cpu_offload: False                                             │
│   enable_layerwise_offload: False                                       │
│                                                                         │
│ 效果: 纯加速，无显存优化                                                 │
└─────────────────────────────────────────────────────────────────────────┘

场景 2: 中等显存压力
┌─────────────────────────────────────────────────────────────────────────┐
│ config:                                                                 │
│   cache_backend: "tea_cache"                                            │
│   enable_cpu_offload: True                                              │
│   enable_layerwise_offload: False                                       │
│                                                                         │
│ 效果: 加速 + Model-Level 显存优化                                       │
└─────────────────────────────────────────────────────────────────────────┘

场景 3: 极端显存压力
┌─────────────────────────────────────────────────────────────────────────┐
│ config:                                                                 │
│   cache_backend: "cache_dit"                                            │
│   enable_cpu_offload: False                                             │
│   enable_layerwise_offload: True                                        │
│                                                                         │
│ 效果: 加速 + Layer-Wise 显存优化                                        │
│ 注意: Layer-Wise 与 Cache-DiT 需要特殊处理                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 九、配置示例

### 9.1 TeaCache 配置

```yaml
# YAML 配置
cache_backend: tea_cache
cache_config:
  rel_l1_thresh: 0.2
```

### 9.2 Cache-DiT 配置

```yaml
# YAML 配置
cache_backend: cache_dit
cache_config:
  Fn_compute_blocks: 1
  max_warmup_steps: 4
  residual_diff_threshold: 0.24
  max_continuous_cached_steps: 3
  enable_taylorseer: false
```

### 9.3 Offloading 配置

```yaml
# Model-Level Offloading
enable_cpu_offload: true
pin_cpu_memory: true

# 或者 Layer-Wise Offloading
enable_layerwise_offload: true
pin_cpu_memory: true
```

---

## 十、总结

### 关键对比

| 维度 | TeaCache | Cache-DiT | Model-Level Offload | Layer-Wise Offload |
|------|----------|-----------|---------------------|-------------------|
| **目的** | 加速 | 加速 | 显存优化 | 显存优化 |
| **粒度** | Transformer | Block | 模型 | Block |
| **实现方式** | Hook | 库集成 | Hook | Hook |
| **性能影响** | +50% 吞吐 | +50-80% 吞吐 | -10-20% 吞吐 | -20-40% 吞吐 |
| **显存节省** | 无 | 无 | ~50% | ~90%+ |
| **适用场景** | 通用加速 | 高级加速 | 中等显存压力 | 极端显存压力 |

### 最佳实践

1. **显存充足**：使用 TeaCache 或 Cache-DiT 加速
2. **中等显存压力**：Model-Level Offload + TeaCache
3. **极端显存压力**：Layer-Wise Offload（可能需要关闭 Cache）
4. **调整阈值**：根据质量/速度权衡调整 `rel_l1_thresh` 或 `residual_diff_threshold`
