# TeaCache 优化实现分析

## 概述

TeaCache (Timestep Embedding Aware Cache) 是一种自适应缓存技术，通过复用相邻 timestep 的 transformer 计算结果来加速扩散模型推理。

---

## 1. 调用流程与触发位置

### 整体调用链路图

```
用户配置 cache_backend="tea_cache"
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  DiffusionModelRunner.load_model()                              │
│  (diffusion_model_runner.py:174-186)                            │
│                                                                 │
│  cache_backend = get_cache_backend("tea_cache", config)         │
│  cache_backend.enable(pipeline)                                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  TeaCacheBackend.enable(pipeline)                               │
│  (backend.py:106-155)                                           │
│                                                                 │
│  1. 创建 TeaCacheConfig (阈值、多项式系数)                        │
│  2. apply_teacache_hook(transformer, config)                    │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  apply_teacache_hook()                                          │
│  (hook.py:255-279)                                              │
│                                                                 │
│  1. HookRegistry.get_or_create(transformer)                     │
│  2. 创建 TeaCacheHook(config)                                   │
│  3. registry.register_hook("teacache", hook)                    │
│     → 替换 transformer 的 forward 方法为 hook.new_forward        │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  每次推理请求                                                    │
│  DiffusionModelRunner.execute_model()                           │
│  (diffusion_model_runner.py:262-268)                            │
│                                                                 │
│  cache_backend.refresh(pipeline, num_steps)                     │
│  → TeaCacheBackend.refresh() → HookRegistry.reset_hook()        │
│  → 重置 TeaCacheState (cnt=0, accumulated_distance=0, etc.)     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  每个 timestep 的 transformer forward                            │
│  TeaCacheHook.new_forward()                                     │
│  (hook.py:87-189)                                               │
│                                                                 │
│  1. extractor_fn(module, *args, **kwargs) → CacheContext        │
│  2. CFG 分支状态管理 (positive/negative)                         │
│  3. _should_compute_full_transformer() → 决定是否计算或复用       │
│  4. 复用缓存残差 或 完整计算并缓存残差                            │
│  5. ctx.postprocess(output)                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 关键触发位置

| 时机 | 位置 | 操作 |
|------|------|------|
| **模型加载时** | `DiffusionModelRunner.load_model():174-186` | 创建 `TeaCacheBackend`，调用 `enable()` 注册 hook |
| **每次生成前** | `DiffusionModelRunner.execute_model():262-268` | 调用 `refresh()` 重置缓存状态 |
| **每个 timestep** | `TeaCacheHook.new_forward()` | 拦截 forward，执行缓存决策逻辑 |

### Hook 机制原理

TeaCache 使用 `HookRegistry` 系统（位于 `vllm_omni/diffusion/hooks/`）：
- `ModelHook` 基类定义了 `new_forward()` 方法
- 注册时，原始 `module.forward` 被替换为 `hook.new_forward`
- 每次 transformer 被调用时，实际执行的是 `TeaCacheHook.new_forward()`

---

## 2. 核心算法逻辑

### 优化原理

扩散模型的去噪过程中，相邻 timestep 的 transformer 输入非常相似。TeaCache 利用这个特性：
1. 比较**调制后输入**的相似度（而非原始 hidden states）
2. 如果相似度高，跳过完整计算，**直接复用上一 timestep 的残差**
3. 残差 = 输出 - 输入，添加残差即可快速得到近似输出

### 算法流程

```python
def _should_compute_full_transformer(state, modulated_input):
    # 步骤 1: 第一个 timestep 总是计算
    if state.cnt == 0:
        state.accumulated_rel_l1_distance = 0.0
        return True

    # 步骤 2: 计算 relative L1 distance
    rel_distance = (
        (modulated_input - state.previous_modulated_input).abs().mean()
        / (state.previous_modulated_input.abs().mean() + 1e-8)
    )

    # 步骤 3: 使用多项式重新缩放（模型特定系数）
    rescaled_distance = polynomial_coefficients(rel_distance)
    state.accumulated_rel_l1_distance += abs(rescaled_distance)

    # 步骤 4: 与阈值比较
    if state.accumulated_rel_l1_distance < threshold:
        return False  # 复用缓存
    else:
        state.accumulated_rel_l1_distance = 0.0  # 重置累加器
        return True  # 完整计算
```

### 缓存命中 vs 未命中

**缓存命中（跳过计算）**:
```python
# Fast Path (hook.py:143-149)
hidden_states = hidden_states + state.previous_residual
encoder_hidden_states = encoder_hidden_states + state.previous_residual_encoder  # 如果有
```

**缓存未命中（完整计算）**:
```python
# Slow Path (hook.py:151-179)
ori_hidden_states = hidden_states.clone()
output = run_transformer_blocks()  # 完整执行所有 transformer blocks

# 缓存残差供下次使用
state.previous_residual = (output - ori_hidden_states).detach()
```

### 多项式系数的作用

不同模型的 embedding 变化特性不同，需要校准的多项式来映射：
- **输入**: modulated input 的 relative L1 distance
- **输出**: 输出 residual 的 relative L1 distance

系数存储在 `config.py` 的 `_MODEL_COEFFICIENTS` 字典中，每个模型有 5 个系数（四次多项式）。

### 阈值与加速比

| rel_l1_thresh | 加速比 | 质量损失 |
|---------------|--------|----------|
| 0.2 (默认) | ~1.5x | 极小 |
| 0.4 | ~1.8x | 轻微 |
| 0.6 | ~2.0x | 可见 |

---

## 3. 类与方法详解

### 3.1 TeaCacheConfig (config.py:81-123)

**作用**: 配置数据类，存储 TeaCache 参数

**属性**:
| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rel_l1_thresh` | float | 0.2 | 累积 L1 距离阈值 |
| `coefficients` | list[float] | None | 多项式系数（None 时自动选择） |
| `transformer_type` | str | "QwenImageTransformer2DModel" | 模型类型标识 |

**生命周期**: 在 `TeaCacheBackend.enable()` 中创建，传递给 `TeaCacheHook`

### 3.2 TeaCacheState (state.py:13-38)

**作用**: 管理单个分支的缓存状态

**属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `cnt` | int | 当前 timestep 计数 |
| `accumulated_rel_l1_distance` | float | 累积的重新缩放 L1 距离 |
| `previous_modulated_input` | Tensor | 上一 timestep 的调制输入 |
| `previous_residual` | Tensor | 上一 timestep 的缓存残差 |
| `previous_residual_encoder` | Tensor | encoder 分支的缓存残差（双流模型） |

**方法**:
- `reset()`: 重置所有状态变量

**生命周期**:
- 创建: `StateManager` 初始化时
- 重置: 每次 `refresh()` 调用时
- 更新: 每个 timestep 的 forward 中

### 3.3 TeaCacheHook (hook.py:30-253)

**作用**: 核心拦截器，实现缓存逻辑

**继承**: `ModelHook` 基类

**属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `config` | TeaCacheConfig | 配置对象 |
| `rescale_func` | np.poly1d | 多项式重缩放函数 |
| `state_manager` | StateManager | 管理多个 CFG 分支的状态 |
| `extractor_fn` | Callable | 模型特定的上下文提取函数 |
| `_forward_cnt` | int | 总 forward 计数（CFG 分支识别） |

**方法**:
| 方法 | 行号 | 说明 |
|------|------|------|
| `initialize_hook()` | 68-85 | 初始化 extractor，设置默认 context |
| `new_forward()` | 87-189 | 拦截 forward，执行缓存逻辑 |
| `_should_compute_full_transformer()` | 191-238 | 核心：决定是否计算或复用 |
| `reset_state()` | 240-252 | 重置所有状态 |

**生命周期**:
- 创建: `apply_teacache_hook()` 中
- 注册: 通过 `HookRegistry.register_hook()`
- 销毁: 随 module 生命周期

### 3.4 CacheContext (extractors.py:30-149)

**作用**: 数据类，封装模型特定的上下文信息

**属性**:
| 属性 | 类型 | 说明 |
|------|------|------|
| `modulated_input` | Tensor | 用于缓存决策的调制输入 |
| `hidden_states` | Tensor | 当前 hidden states |
| `encoder_hidden_states` | Tensor \| None | encoder states（双流模型） |
| `temb` | Tensor | 时间步嵌入 |
| `run_transformer_blocks` | Callable | 执行 transformer blocks 的函数 |
| `postprocess` | Callable | 后处理函数 |
| `extra_states` | dict \| None | 额外状态（如 Flux2 的 single blocks） |

**方法**:
- `validate()`: 验证上下文数据的有效性

**生命周期**: 每次 forward 调用时创建，用完即弃

### 3.5 TeaCacheBackend (backend.py:86-196)

**作用**: 实现 `CacheBackend` 接口，管理 TeaCache 生命周期

**方法**:
| 方法 | 行号 | 说明 |
|------|------|------|
| `enable(pipeline)` | 106-155 | 应用 TeaCache hook 到 transformer |
| `refresh(pipeline, num_steps)` | 157-196 | 重置状态用于新生成 |
| `is_enabled()` | 继承 | 检查是否启用 |

**生命周期**:
- 创建: `DiffusionModelRunner.load_model()` 中
- 启用: `load_model()` 结束时调用 `enable()`
- 刷新: 每次 `execute_model()` 开始时

### 3.6 Extractor 函数 (extractors.py)

**作用**: 模型特定的上下文提取器，封装所有模型差异

**注册表**: `EXTRACTOR_REGISTRY` (1202-1213 行)

**已支持的模型**:
| 函数名 | 模型类型 |
|--------|----------|
| `extract_qwen_context` | QwenImageTransformer2DModel |
| `extract_flux_context` | FluxTransformer2DModel |
| `extract_flux2_context` | Flux2Transformer2DModel |
| `extract_flux2_klein_context` | Flux2Klein |
| `extract_bagel_context` | Bagel |
| `extract_zimage_context` | ZImageTransformer2DModel |
| `extract_longcat_context` | LongCatImageTransformer2DModel |
| `extract_stable_audio_context` | StableAudioDiTModel |

**每个 extractor 的职责**:
1. **预处理**: 模型特定的输入处理（embed, norm 等）
2. **提取调制输入**: 从第一个 transformer block 获取调制后的 hidden states
3. **定义执行函数**: `run_transformer_blocks()` 可调用对象
4. **定义后处理**: `postprocess()` 可调用对象
5. **返回 CacheContext**

---

## 4. 添加新模型支持

### 需要做的工作

#### 步骤 1: 创建 Extractor 函数

在 `extractors.py` 中添加：

```python
def extract_new_model_context(
    module: nn.Module,
    *args,
    **kwargs
) -> CacheContext:
    """
    Extract cache context for NewModelTransformer.

    必须完成的步骤:
    1. 模型特定预处理（与原始 forward 一致）
    2. 提取 modulated_input（第一个 block 的调制输出）
    3. 定义 run_transformer_blocks()
    4. 定义 postprocess()
    5. 返回 CacheContext
    """
    # 1. 预处理（复制原始 forward 的预处理逻辑）
    hidden_states = module.x_embedder(hidden_states)
    temb = module.time_embed(timestep)
    encoder_hidden_states = module.context_embedder(encoder_hidden_states)

    # 2. 提取调制输入（关键！）
    first_block = module.transformer_blocks[0]
    modulated_input = first_block.norm1(hidden_states, emb=temb)  # 模型特定

    # 3. 定义 transformer 执行
    def run_transformer_blocks():
        h = hidden_states
        e = encoder_hidden_states
        for block in module.transformer_blocks:
            h, e = block(h, e, temb=temb, ...)  # 模型特定参数
        return (h, e)

    # 4. 定义后处理
    def postprocess(h):
        h = module.norm_out(h, temb)
        return module.proj_out(h)

    # 5. 返回上下文
    return CacheContext(
        modulated_input=modulated_input,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        run_transformer_blocks=run_transformer_blocks,
        postprocess=postprocess,
    )
```

#### 步骤 2: 注册 Extractor

方式 A - 在 `EXTRACTOR_REGISTRY` 中添加（推荐）:
```python
# extractors.py 末尾
EXTRACTOR_REGISTRY["NewModelTransformer"] = extract_new_model_context
```

方式 B - 使用注册函数（运行时动态注册）:
```python
from vllm_omni.diffusion.cache.teacache import register_extractor
register_extractor("NewModelTransformer", extract_new_model_context)
```

#### 步骤 3: 添加多项式系数

在 `config.py` 的 `_MODEL_COEFFICIENTS` 字典中添加：
```python
_MODEL_COEFFICIENTS = {
    # ... 现有模型 ...
    "NewModelTransformer": [
        # 使用 coefficient_estimator.py 估计，或借用相似模型的系数
        c4, c3, c2, c1, c0  # 四次多项式系数
    ],
}
```

**系数估计方法**:
1. 使用 `coefficient_estimator.py` 工具
2. 收集不同 timestep 的输入 L1 distance 和输出 residual L1 distance
3. 使用 `np.polyfit()` 拟合

#### 步骤 4: 测试验证

```python
# 单元测试
def test_new_model_extractor():
    transformer = NewModelTransformer(...)
    extractor = get_extractor("NewModelTransformer")
    ctx = extractor(transformer, hidden_states, timestep, ...)
    ctx.validate()  # 验证上下文有效性

    # 验证缓存逻辑
    assert ctx.run_transformer_blocks() is not None
    assert ctx.postprocess(hidden_states) is not None
```

### 特殊情况处理

#### 如果模型使用自定义 Pipeline enabler

某些模型（如 HunyuanImage3）不适合使用 hook 方式，需要在 `backend.py` 中添加自定义 enabler：

```python
def enable_new_model_teacache(pipeline, config):
    # 特殊处理逻辑
    pipeline._tea_cache_config = TeaCacheConfig(...)
    # 或其他模型特定的初始化

CUSTOM_TEACACHE_ENABLERS = {
    # ... 现有 ...
    "NewModelPipeline": enable_new_model_teacache,
}
```

---

## 文件结构总结

```
vllm_omni/diffusion/cache/teacache/
├── __init__.py          # 公开 API 导出
├── config.py            # TeaCacheConfig + 模型特定多项式系数
├── state.py             # TeaCacheState 状态管理
├── backend.py           # TeaCacheBackend 实现 CacheBackend 接口
├── hook.py              # TeaCacheHook 核心拦截逻辑
├── extractors.py        # CacheContext + 模型特定提取器
└── coefficient_estimator.py  # 多项式系数估计工具
```

---

## 关键设计模式

1. **Hook 模式**: 通过 `ModelHook` 拦截 forward，零侵入模型代码
2. **Extractor 模式**: 模型特定逻辑封装在 extractor 中，核心缓存逻辑保持通用
3. **Context 模式**: `CacheContext` 统一不同模型的上下文接口
4. **Registry 模式**: `EXTRACTOR_REGISTRY` 支持动态扩展新模型
