# Diffusers 模型加载方式与 Registry 机制详解

## 一、模型加载方式对比

在 `diffusers_loader.py` 的 `load_model` 方法中，共有 **4 种** 加载模型的方式：

### 1. `initialize_model(od_config)` - **默认方式**

```python
model = initialize_model(od_config)
```

**作用**：使用 vLLM 自定义的 Pipeline 类加载模型。

**特点**：
- 通过 `DiffusionModelRegistry` 注册表查找对应的 Pipeline 类
- 使用 vLLM 优化的权重加载逻辑
- 支持量化、序列并行等高级特性
- 权重从 safetensors/bin 文件加载

**适用场景**：大多数情况下的默认选择，性能最优。

---

### 2. `DiffusersAdapterPipeline` - **Diffusers 兼容方式**

```python
model = DiffusersAdapterPipeline(od_config=od_config, device=target_device)
```

**作用**：使用 HuggingFace diffusers 原生的 `DiffusionPipeline.from_pretrained()` 加载模型。

**解决的问题**：
- 支持那些尚未在 vLLM 中实现自定义 Pipeline 的模型
- 快速接入新模型，无需编写适配代码
- 兼容 diffusers 生态的所有组件

**不同点**：
- 使用 diffusers 的权重加载逻辑，而非 vLLM 自定义的
- 不支持 vLLM 特有的优化（如自定义量化、序列并行等）
- 适合快速验证或作为过渡方案

---

### 3. `custom_pipeline` - **自定义 Pipeline 方式**

```python
model_cls = resolve_obj_by_qualname(custom_pipeline_name)
model = model_cls(od_config=od_config)
```

**作用**：通过全限定名动态加载用户自定义的 Pipeline 类。

**解决的问题**：
- 允许用户在不修改代码的情况下注入自定义 Pipeline
- 支持实验性的或私有模型实现
- 提供扩展点，无需修改 registry

**不同点**：
- 跳过注册表查找，直接通过 qualname 加载
- `custom_pipeline_name` 格式：`module.path.ClassName`
- 例如：`my_module.pipelines.MyCustomPipeline`

---

### 4. HSDP 模式 - **分布式推理方式**

```python
model = self._load_model_with_hsdp(od_config, load_format=load_format, ...)
```

**作用**：为混合分片数据并行推理加载模型。

**解决的问题**：
- 大模型单卡放不下时，通过 HSDP 将 transformer 分片到多卡
- 支持 MoE 模型的两阶段 transformer（如 Wan2.2-I2V 的 `transformer_2`）

**不同点**：
- 权重先加载到 CPU，再通过 `apply_hsdp_to_model` 重新分布
- 只对 transformer 组件进行分片，VAE、text_encoder 等组件正常加载
- 需要配置 `parallel_config.use_hsdp=True`

---

## 二、Registry 注册机制详解

### 2.1 注册表数据结构

注册表定义在 `vllm_omni/diffusion/registry.py`：

```python
_DIFFUSION_MODELS = {
    # arch_name: (mod_folder, mod_relname, cls_name)
    "QwenImagePipeline": (
        "qwen_image",           # 模块文件夹名
        "pipeline_qwen_image",  # 文件名（不含.py）
        "QwenImagePipeline",    # 类名
    ),
    "WanPipeline": (
        "wan2_2",
        "pipeline_wan2_2",
        "Wan22Pipeline",
    ),
    # ... 更多模型
}
```

**映射关系**：
```
model_class_name -> (mod_folder, mod_relname, cls_name)
                        ↓
    vllm_omni.diffusion.models.{mod_folder}.{mod_relname}
                        ↓
                  导入 cls_name 类
```

例如 `WanPipeline` → `vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.Wan22Pipeline`

---

### 2.2 Registry 创建

```python
DiffusionModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm_omni.diffusion.models.{mod_folder}.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_folder, mod_relname, cls_name) in _DIFFUSION_MODELS.items()
    }
)
```

**关键点**：
- `_ModelRegistry` 来自 vLLM 核心库
- `_LazyRegisteredModel` 实现延迟加载，只有在真正需要时才导入模块

---

### 2.3 `_try_load_model_cls` 实现逻辑

该方法定义在 vLLM 核心库中：

```python
# 位置：vllm/model_executor/models/registry.py

@lru_cache(maxsize=128)
def _try_load_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> type[nn.Module] | None:
    from vllm.platforms import current_platform

    current_platform.verify_model_arch(model_arch)  # 验证模型架构
    try:
        return model.load_model_cls()  # 调用延迟加载模型的加载方法
    except Exception:
        logger.exception("Error in loading model architecture '%s'", model_arch)
        return None
```

**`load_model_cls` 实现**：

```python
class _LazyRegisteredModel:
    module_name: str   # 例如 "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2"
    class_name: str    # 例如 "Wan22Pipeline"

    def load_model_cls(self) -> type[nn.Module]:
        mod = importlib.import_module(self.module_name)  # 动态导入模块
        return getattr(mod, self.class_name)            # 获取类对象
```

---

### 2.4 完整调用链

```
initialize_model(od_config)
    │
    ├─► DiffusionModelRegistry._try_load_model_cls(od_config.model_class_name)
    │       │
    │       ├─► 从 registry.models 字典中获取 _LazyRegisteredModel
    │       │
    │       └─► model.load_model_cls()
    │               │
    │               ├─► importlib.import_module("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2")
    │               │
    │               └─► getattr(module, "Wan22Pipeline")
    │
    └─► model_class(od_config=od_config)  # 实例化 Pipeline
```

---

## 三、如何添加新模型

### 方法 1：修改 Registry（推荐）

在 `_DIFFUSION_MODELS` 字典中添加条目：

```python
"MyNewPipeline": (
    "my_new_model",           # 创建文件夹 vllm_omni/diffusion/models/my_new_model/
    "pipeline_my_new_model",  # 创建文件 pipeline_my_new_model.py
    "MyNewPipeline",          # 在该文件中定义类 MyNewPipeline
),
```

### 方法 2：使用 custom_pipeline

启动时传入参数：
```python
load_model(
    od_config,
    load_format="custom_pipeline",
    custom_pipeline_name="my_module.my_pipeline.MyPipeline"
)
```

---

## 四、相关文件位置

| 文件 | 作用 |
|------|------|
| `vllm_omni/diffusion/registry.py` | Diffusion 模型注册表定义 |
| `vllm_omni/diffusion/model_loader/diffusers_loader.py` | 模型加载器实现 |
| `vllm_omni/diffusion/models/<mod_folder>/` | 具体 Pipeline 实现目录 |
| `vllm/model_executor/models/registry.py` | vLLM 核心 Registry 实现 |
