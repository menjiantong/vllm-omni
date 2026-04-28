# FP8 量化全链路指南

## 目录

1. [概述](#1-概述)
2. [FP8 数据格式](#2-fp8-数据格式)
3. [完整调用链图解](#3-完整调用链图解)
4. [阶段一：用户调用入口](#4-阶段一用户调用入口)
5. [阶段二：引擎初始化](#5-阶段二引擎初始化)
6. [阶段三：配置构建](#6-阶段三配置构建)
7. [阶段四：模型初始化](#7-阶段四模型初始化)
8. [阶段五：权重加载与量化](#8-阶段五权重加载与量化)
9. [阶段六：推理计算](#9-阶段六推理计算)
10. [FP8 量化核心实现](#10-fp8-量化核心实现)
11. [示例：FLUX.2-klein FP8 量化](#11-示例flux2-klein-fp8-量化)

---

## 1. 概述

FP8（8-bit Floating Point）量化是一种将 BF16/FP16 权重压缩为 8-bit 浮点数的技术，可将模型显存占用减少约 50%。

### 量化模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| **在线量化（Online）** | 加载 BF16/FP16 检查点时，自动转换为 FP8 | 快速部署，无需预处理 |
| **离线量化（Offline）** | 加载已预先量化的 FP8 检查点 | 生产环境，最佳性能 |

---

## 2. FP8 数据格式

### E4M3FN 格式

```
┌───┬───────┬───────────┐
│ S │ E[3:0]│ M[2:0]    │
└───┴───────┴───────────┘
 1bit  4bits    3bits

数值范围: ±[2^-6, 448]
精度: 约 3-4 位有效数字
```

### 量化公式

```python
# 量化
scale = max_abs_weight / 448.0
weight_fp8 = round(weight / scale).clamp(-448, 447).to(torch.float8_e4m3fn)

# 反量化
weight_bf16 = weight_fp8.to(torch.bfloat16) * scale
```

---

## 3. 完整调用链图解

### 总体架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              【用户层】                                       │
│                                                                              │
│   Omni(model="FLUX.2-klein-4B", quantization="fp8")                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           【入口层 - 非量化】                                 │
│                                                                              │
│   OmniBase.__init__()                                                        │
│       │                                                                      │
│       ├──► omni_snapshot_download(model)     # 下载模型                      │
│       │                                                                      │
│       └──► AsyncOmniEngine(model, **kwargs)  # 创建引擎                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         【引擎初始化层 - 非量化】                             │
│                                                                              │
│   AsyncOmniEngine.__init__()                                                 │
│       │                                                                      │
│       ├──► _resolve_stage_configs()          # 解析阶段配置                  │
│       │                                                                      │
│       ├──► _bootstrap_orchestrator()         # 启动编排器线程               │
│       │                                                                      │
│       └──► initialize_diffusion_stage()      # 初始化扩散阶段               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          【配置构建层 - 量化关键】                            │
│                                                                              │
│   build_diffusion_config(model, stage_cfg, metadata)                        │
│       │                                                                      │
│       └──► OmniDiffusionConfig.from_kwargs(                                 │
│               model=model,                                                   │
│               quantization="fp8",  ← 用户传入的量化参数                      │
│               ...                                                            │
│           )                                                                  │
│               │                                                              │
│               ├──► 参数兼容性处理                                            │
│               │   "quantization" → "quantization_config"                     │
│               │                                                              │
│               └──► 【量化核心】build_quant_config("fp8")                     │
│                       │                                                      │
│                       └──► Fp8Config(                                        │
│                               is_checkpoint_fp8_serialized=False,            │
│                               activation_scheme="dynamic"                    │
│                           )                                                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       【Diffusion Client 创建 - 非量化】                      │
│                                                                              │
│   create_diffusion_client(model, od_config, metadata, ...)                  │
│       │                                                                      │
│       └──► StageDiffusionClient.__init__()                                   │
│               │                                                              │
│               ├──► spawn_diffusion_proc()     # 启动子进程                   │
│               │                                                              │
│               └──► complete_diffusion_handshake()  # 握手确认               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        【Diffusion 进程初始化 - 非量化】                      │
│                                                                              │
│   StageDiffusionProc.initialize()                                            │
│       │                                                                      │
│       ├──► _enrich_config()                   # 加载模型元数据               │
│       │                                                                      │
│       └──► DiffusionEngine.make_engine(od_config)                           │
│               │                                                              │
│               ├──► DiffusionExecutor.get_class()  # 获取执行器类            │
│               │                                                              │
│               └──► MultiprocDiffusionExecutor._init_executor()              │
│                       │                                                      │
│                       └──► _launch_workers()    # 启动 Worker 进程          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          【Worker 初始化 - 非量化】                           │
│                                                                              │
│   DiffusionWorker.__init__()                                                 │
│       │                                                                      │
│       ├──► init_device()                      # 初始化设备和分布式环境       │
│       │       ├──► torch.cuda.set_device()                                   │
│       │       ├──► init_distributed_environment()                            │
│       │       └──► initialize_model_parallel()                               │
│       │                                                                      │
│       ├──► DiffusionModelRunner.__init__()    # 创建模型运行器              │
│       │                                                                      │
│       └──► load_model()                       # 加载模型 ← 进入下一阶段     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        【模型加载层 - 量化关键】                              │
│                                                                              │
│   DiffusionModelRunner.load_model()                                          │
│       │                                                                      │
│       └──► DiffusersPipelineLoader.load_model(od_config)                    │
│               │                                                              │
│               ├──► 【量化关键】initialize_model(od_config)                  │
│               │       │                                                      │
│               │       ├──► _prepare_diffusion_quant_config()                │
│               │       │       │                                              │
│               │       │       └──► configure_quant_config()                 │
│               │       │               作用: 注入 packed_modules_mapping     │
│               │       │                                                      │
│               │       └──► Flux2KleinPipeline(od_config=od_config)          │
│               │               │                                              │
│               │               └──► Transformer 初始化                        │
│               │                       │                                      │
│               │                       └──► LinearBase.__init__()            │
│               │                               │                              │
│               │                               └──► 【量化核心】              │
│               │                                   quant_config              │
│               │                                   .get_quant_method()        │
│               │                                       │                      │
│               │                                       └──► Fp8OnlineLinear   │
│               │                                           Method           │
│               │                                                              │
│               ├──► load_weights(model)         # 加载 BF16 权重             │
│               │       作用: 从 HuggingFace 下载并加载权重到模型              │
│               │                                                              │
│               └──► 【量化核心】_process_weights_after_loading(model)         │
│                       │                                                      │
│                       └──► for module in model.modules():                   │
│                               module.quant_method                            │
│                                   .process_weights_after_loading()           │
│                                       │                                      │
│                                       └──► ops.scaled_fp8_quant()           │
│                                               BF16 → FP8 转换                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            【推理层 - 量化关键】                              │
│                                                                              │
│   omni.generate(prompt)                                                      │
│       │                                                                      │
│       ├──► 文本编码（非量化）                                                │
│       │       prompt → T5 text_encoder → encoder_hidden_states              │
│       │                                                                      │
│       ├──► 初始噪声（非量化）                                                │
│       │       latents = randn(shape)                                         │
│       │                                                                      │
│       ├──► 去噪循环（非量化框架，量化计算）                                  │
│       │       for t in timesteps:                                            │
│       │           transformer.forward(latents, t, hidden_states)            │
│       │               │                                                      │
│       │               └──► for block in transformer_blocks:                 │
│       │                       │                                              │
│       │                       ├──► attention(hidden_states)                  │
│       │                       │       │                                      │
│       │                       │       └──► to_qkv(x)                         │
│       │                       │               │                              │
│       │                       │               └──► 【量化核心】              │
│       │                       │                   Fp8OnlineLinearMethod      │
│       │                       │                   .apply()                   │
│       │                       │                       │                      │
│       │                       │                       ├──► 动态量化输入      │
│       │                       │                       ├──► FP8 GEMM          │
│       │                       │                       └──► 反量化输出        │
│       │                       │                                              │
│       │                       └──► mlp(hidden_states)  # 同上                │
│       │                                                                      │
│       └──► VAE 解码（非量化）                                                │
│               latents → vae.decode() → image                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 阶段一：用户调用入口

### 4.1 入口类调用

**用户代码**:
```python
from vllm_omni import Omni

omni = Omni(model="black-forest-labs/FLUX.2-klein-4B", quantization="fp8")
outputs = omni.generate("a cat")
```

### 4.2 Omni 类

**文件**: `vllm_omni/entrypoints/omni.py`

```python
class Omni(OmniBase):
    """同步离线生成的入口类"""
    
    # Omni 继承自 OmniBase，主要逻辑在 OmniBase.__init__() 中
```

### 4.3 OmniBase 初始化

**文件**: `vllm_omni/entrypoints/omni_base.py`

```python
class OmniBase:
    def __init__(self, model: str, **kwargs):
        # ========== 非量化步骤 ==========
        # 1. 下载模型（如果需要）
        model = omni_snapshot_download(model)
        self.model = model
        
        # 2. 创建异步引擎
        self.engine = AsyncOmniEngine(
            model=model,
            **kwargs,  # quantization="fp8" 在这里传入
        )
```

**作用**: 
- `omni_snapshot_download()`: 检查模型是否已缓存，未缓存则从 HuggingFace 下载
- `AsyncOmniEngine`: 创建多阶段推理引擎

---

## 5. 阶段二：引擎初始化

### 5.1 AsyncOmniEngine 初始化

**文件**: `vllm_omni/engine/async_omni_engine.py`

```python
class AsyncOmniEngine:
    def __init__(
        self,
        model: str,
        stage_init_timeout: int = 300,
        init_timeout: int = 600,
        **kwargs,
    ):
        self.model = model
        
        # ========== 非量化步骤 ==========
        # 1. 解析阶段配置
        self.config_path, self.stage_configs = self._resolve_stage_configs(model, kwargs)
        self.num_stages = len(self.stage_configs)
        
        # 2. 启动编排器线程
        self.orchestrator_thread = threading.Thread(
            target=self._bootstrap_orchestrator,
            args=(stage_init_timeout, startup_future),
            daemon=True,
        )
        self.orchestrator_thread.start()
        
        # 3. 等待初始化完成
        self._wait_for_orchestrator_init(startup_future, startup_timeout)
```

**作用**:
- `_resolve_stage_configs()`: 解析多阶段配置（如 LLM + Diffusion）
- `_bootstrap_orchestrator()`: 在后台线程启动编排器，管理各阶段执行

### 5.2 扩散阶段初始化

**文件**: `vllm_omni/engine/stage_init_utils.py`

```python
def initialize_diffusion_stage(
    stage_id: int,
    model: str,
    stage_cfg: Any,
    metadata: StageMetadata,
    stage_init_timeout: int,
    batch_size: int = 1,
) -> Any:
    """初始化扩散阶段"""
    
    # ========== 非量化步骤 ==========
    # 1. 构建配置
    engine_args = _to_dict(stage_cfg.engine_args)
    
    # ========== 进入配置构建（下一阶段）==========
    od_config = OmniDiffusionConfig.from_kwargs(
        stage_id=stage_id,
        model=model,
        **engine_args,  # quantization="fp8" 在这里
    )
    
    # 2. 完整配置构建
    od_config = build_diffusion_config(model, stage_cfg, metadata)
    
    # 3. 创建 Diffusion Client
    return create_diffusion_client(model, od_config, metadata, ...)
```

**作用**:
- 构建 `OmniDiffusionConfig`，包含量化配置
- 创建 `StageDiffusionClient`，与子进程通信

---

## 6. 阶段三：配置构建

### 6.1 OmniDiffusionConfig.from_kwargs

**文件**: `vllm_omni/diffusion/data.py`

```python
class OmniDiffusionConfig:
    quantization_config: str | QuantizationConfig | dict | None = None
    
    @classmethod
    def from_kwargs(cls, **kwargs) -> "OmniDiffusionConfig":
        # ========== 非量化步骤：参数兼容性处理 ==========
        # 向后兼容：quantization → quantization_config
        if "quantization" in kwargs and kwargs.get("quantization_config") is None:
            kwargs["quantization_config"] = kwargs.pop("quantization")
        
        # 过滤无效字段
        valid_fields = {f.name for f in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        
        # 创建配置对象
        config = cls(**filtered_kwargs)
        
        # ========== 量化核心：解析量化配置 ==========
        # 在 __post_init__ 中处理
        return config
    
    def __post_init__(self):
        # ... 其他初始化 ...
        
        # ========== 量化核心步骤 ==========
        # 解析 quantization_config
        if self.quantization_config is not None:
            if isinstance(self.quantization_config, str):
                # "fp8" → Fp8Config
                self.quantization_config = build_quant_config(self.quantization_config)
            elif isinstance(self.quantization_config, Mapping):
                # {"method": "fp8", ...} → Fp8Config
                self.quantization_config = build_quant_config(dict(self.quantization_config))
            # 如果已经是 QuantizationConfig，直接使用
```

**作用**:
- 参数兼容性处理（旧参数名映射）
- 调用 `build_quant_config()` 将字符串/字典转换为 `Fp8Config` 对象

### 6.2 build_quant_config

**文件**: `vllm_omni/quantization/factory.py`

```python
def build_quant_config(
    spec: str | dict | QuantizationConfig | None,
    **kwargs,
) -> QuantizationConfig | None:
    """构建量化配置
    
    支持格式:
    - "fp8"                          # 字符串
    - {"method": "fp8"}              # 字典
    - {"transformer": "fp8"}         # 组件级
    - Fp8Config()                    # 已构建对象
    """
    if spec is None:
        return None
    
    if isinstance(spec, QuantizationConfig):
        return spec  # 已经是配置对象，直接返回
    
    if isinstance(spec, str):
        return _build_single(spec, **kwargs)
    
    if isinstance(spec, Mapping):
        # 检查是否为组件级配置
        if _is_per_component_dict(spec):
            return _build_component_config(spec)
        method = spec.pop("method", None)
        return _build_single(method, **spec, **kwargs)


def _build_single(method: str, **kwargs) -> QuantizationConfig:
    """构建单个量化配置"""
    method = method.lower()
    
    # 检查是否有自定义覆盖
    if method in _OVERRIDES:
        return _OVERRIDES[method](**kwargs)
    
    # FP8 使用 vLLM 的配置类
    config_cls = get_quantization_config(method)
    
    # ========== 量化核心：创建 Fp8Config ==========
    return config_cls(**kwargs)
```

**作用**:
- 统一的量化配置构建入口
- 将字符串 "fp8" 转换为 `Fp8Config` 对象

### 6.3 Fp8Config 类

**文件**: `vllm/model_executor/layers/quantization/fp8.py`

```python
class Fp8Config(QuantizationConfig):
    """FP8 量化配置"""
    
    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,  # 是否预量化检查点
        activation_scheme: str = "dynamic",           # 激活量化方案
        ignored_layers: list[str] | None = None,      # 跳过的层
        weight_block_size: list[int] | None = None,   # 块量化大小
    ):
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.weight_block_size = weight_block_size
    
    # ========== 量化核心方法 ==========
    def get_quant_method(
        self, 
        layer: torch.nn.Module, 
        prefix: str
    ) -> QuantizeMethodBase | None:
        """为指定层分配合适的量化方法
        
        这是连接配置和实际量化的桥梁！
        
        参数:
            layer: 当前层对象（LinearBase 或 FusedMoE）
            prefix: 层路径，如 "transformer.blocks.0.attn.to_qkv"
        
        返回:
            量化方法对象，或 None（不量化）
        """
        if isinstance(layer, LinearBase):
            # 检查是否跳过该层
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            
            # 根据检查点类型选择方法
            if not self.is_checkpoint_fp8_serialized:
                # 在线量化：BF16 → FP8
                return Fp8OnlineLinearMethod(self)
            else:
                # 离线量化：已经是 FP8
                return Fp8LinearMethod(self)
        
        elif isinstance(layer, FusedMoE):
            # MoE 专家层量化
            if self.is_checkpoint_fp8_serialized:
                return Fp8MoEMethod(self, layer)
            else:
                return Fp8OnlineMoEMethod(self, layer)
        
        return None  # 不支持量化的层
```

**关键属性**:
- `is_checkpoint_fp8_serialized`: 决定是在线量化还是离线量化
- `activation_scheme`: "dynamic"（动态）或 "static"（静态）
- `ignored_layers`: 跳过量化的层名模式

**关键方法**:
- `get_quant_method()`: **核心桥梁**，为每个 Linear 层分配量化方法

---

## 7. 阶段四：模型初始化

### 7.1 DiffusionModelRunner 初始化

**文件**: `vllm_omni/diffusion/worker/diffusion_model_runner.py`

```python
class DiffusionModelRunner:
    def __init__(self, vllm_config, od_config, device):
        self.od_config = od_config  # 包含 quantization_config
        self.device = device
        self.pipeline = None
    
    def load_model(self, ...):
        # 创建模型加载器
        model_loader = DiffusersPipelineLoader(
            load_config, 
            od_config=self.od_config
        )
        
        # ========== 加载模型（进入下一阶段）==========
        self.pipeline = model_loader.load_model(
            od_config=self.od_config,
            load_device=load_device,
            ...
        )
```

**作用**: 管理模型加载、编译和缓存

### 7.2 DiffusersPipelineLoader.load_model

**文件**: `vllm_omni/diffusion/model_loader/diffusers_loader.py`

```python
class DiffusersPipelineLoader:
    def load_model(self, od_config, load_device, ...):
        target_device = torch.device(load_device)
        
        with set_default_torch_dtype(od_config.dtype):
            # ========== 非量化步骤 ==========
            # 检查 HSDP 配置
            
            # ========== 量化关键：初始化模型 ==========
            model = initialize_model(od_config)
            
            # ========== 非量化步骤 ==========
            # 加载权重
            self.load_weights(model)
            
            # ========== 量化关键：处理权重 ==========
            self._process_weights_after_loading(model, target_device)
        
        return model.eval()
```

**作用**:
- 协调模型初始化、权重加载、量化处理

### 7.3 initialize_model

**文件**: `vllm_omni/diffusion/registry.py`

```python
def initialize_model(od_config: OmniDiffusionConfig) -> nn.Module:
    """初始化扩散模型"""
    
    # 从注册表获取模型类
    model_class = DiffusionModelRegistry._try_load_model_cls(
        od_config.model_class_name
    )
    
    if model_class is not None:
        # ========== 量化关键：准备量化配置 ==========
        _prepare_diffusion_quant_config(od_config, model_class)
        
        # ========== 实例化模型 ==========
        model = model_class(od_config=od_config)
        
        # ... 其他初始化 ...
        
        return model


def _prepare_diffusion_quant_config(
    od_config: OmniDiffusionConfig,
    model_class: type[nn.Module],
) -> None:
    """准备扩散模型的量化配置"""
    quant_config = od_config.quantization_config
    if quant_config is None:
        return
    
    # 更新配置（如果需要）
    if hasattr(quant_config, "maybe_update_config"):
        quant_config.maybe_update_config(od_config.model)
    
    # 注入 packed_modules_mapping（用于 QKV 融合层）
    diffusion_packed_modules_mapping = current_omni_platform.get_diffusion_packed_modules_mapping(model_class)
    if diffusion_packed_modules_mapping is not None:
        model_class.packed_modules_mapping = diffusion_packed_modules_mapping
    
    # vLLM 配置函数
    configure_quant_config(quant_config, model_class)
```

**作用**:
- 从注册表获取模型类（如 `Flux2KleinPipeline`）
- 准备量化配置，注入融合层映射

### 7.4 Pipeline 初始化

**文件**: `vllm_omni/diffusion/models/flux2_klein/pipeline_flux2_klein.py`

```python
class Flux2KleinPipeline:
    def __init__(self, od_config: OmniDiffusionConfig):
        # ========== 非量化步骤 ==========
        # 初始化 text_encoder (T5)
        self.text_encoder = ...
        
        # 初始化 VAE
        self.vae = ...
        
        # ========== 量化关键：初始化 Transformer ==========
        self.transformer = Flux2KleinTransformer2DModel(
            quant_config=od_config.quantization_config,  # Fp8Config
            ...
        )
```

**作用**:
- 初始化各组件，量化配置传递给 Transformer

### 7.5 Transformer 初始化

**文件**: `vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py`

```python
class Flux2KleinTransformer2DModel:
    def __init__(self, quant_config, ...):
        # 初始化各层
        for i in range(num_layers):
            # ========== 量化关键：Linear 层初始化 ==========
            self.transformer_blocks[i] = Flux2KleinBlock(
                quant_config=quant_config,  # Fp8Config
                ...
            )
```

### 7.6 LinearBase 初始化

**文件**: `vllm/model_executor/layers/linear.py`

```python
class LinearBase(PluggableLayer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        quant_config: QuantizationConfig | None = None,  # Fp8Config
        prefix: str = "",  # 层路径
        ...
    ):
        self.quant_config = quant_config
        self.prefix = prefix
        
        # ========== 量化核心：获取量化方法 ==========
        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        else:
            # 调用 Fp8Config.get_quant_method()
            # 返回 Fp8OnlineLinearMethod 实例
            self.quant_method = quant_config.get_quant_method(self, prefix)
        
        # 创建权重占位符
        self.quant_method.create_weights(self, ...)
```

**作用**:
- 每个 Linear 层初始化时获取对应的量化方法
- `quant_method` 将在权重加载和推理时使用

### 7.7 Fp8OnlineLinearMethod.create_weights

**文件**: `vllm/model_executor/layers/quantization/fp8.py`

```python
class Fp8OnlineLinearMethod(Fp8LinearMethod):
    uses_meta_device: bool = True  # 支持延迟加载
    
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """创建 FP8 权重占位符"""
        
        output_size_per_partition = sum(output_partition_sizes)
        
        # 保存元信息
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype  # BF16
        layer.weight_block_size = None
        
        # ========== 创建权重占位符（meta device）==========
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",  # 延迟初始化，不占显存
                dtype=params_dtype,  # BF16
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        
        # 设置延迟处理标志
        initialize_online_processing(layer)
```

**作用**:
- 创建 `meta` device 上的权重占位符
- 设置延迟处理标志，等待实际权重加载

---

## 8. 阶段五：权重加载与量化

### 8.1 load_weights

**文件**: `vllm_omni/diffusion/model_loader/diffusers_loader.py`

```python
def load_weights(self, model: nn.Module) -> None:
    """加载模型权重"""
    
    # 获取需要加载的权重名
    weights_to_load = self._get_expected_parameter_names(model)
    
    # ========== 非量化步骤 ==========
    # 从 HuggingFace 加载权重
    loaded_weights = model.load_weights(self.get_all_weights(model))
    
    # 此时，layer.weight 是 BF16 权重
```

**作用**: 从 HuggingFace 下载并加载 BF16 权重到模型

### 8.2 _process_weights_after_loading

**文件**: `vllm_omni/diffusion/model_loader/diffusers_loader.py`

```python
def _process_weights_after_loading(
    self, 
    model: nn.Module, 
    target_device: torch.device
) -> None:
    """权重加载后处理（量化）"""
    
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        
        if isinstance(quant_method, QuantizeMethodBase):
            # 移动到目标设备
            module_device = next(module.parameters(), None)
            if module_device is not None:
                module_device = module_device.device
            needs_device_move = module_device != target_device
            
            if needs_device_move:
                module.to(target_device)
            
            # ========== 量化核心：执行量化 ==========
            quant_method.process_weights_after_loading(module)
            
            if needs_device_move:
                module.to(module_device)
```

**作用**: 遍历所有模块，调用量化方法处理权重

### 8.3 Fp8OnlineLinearMethod.process_weights_after_loading

**文件**: `vllm/model_executor/layers/quantization/fp8.py`

```python
class Fp8OnlineLinearMethod(Fp8LinearMethod):
    
    def process_weights_after_loading(self, layer: Module) -> None:
        """执行 FP8 量化
        
        这是量化的核心步骤！
        
        输入: layer.weight (BF16)
        输出: layer.weight (FP8) + layer.weight_scale
        """
        # 防止重复处理
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        
        assert not self.block_quant  # 在线量化不支持块量化
        layer.input_scale = None
        
        # ========== 量化核心：调用 CUDA kernel ==========
        qweight, weight_scale = ops.scaled_fp8_quant(
            layer.weight,  # BF16 输入
            scale=None     # 自动计算 scale
        )
        # qweight: torch.float8_e4m3fn
        # weight_scale: torch.float32 标量
        
        # 替换权重
        replace_parameter(layer, "weight", qweight.data)
        replace_parameter(layer, "weight_scale", weight_scale.data)
        
        # 转置权重以适配 kernel
        weight = qweight.t()
        replace_parameter(layer, "weight", weight.data)
        
        # 标记已处理
        layer._already_called_process_weights_after_loading = True
```

### 8.4 量化前后对比

```
量化前:
┌─────────────────────────────────────────────┐
│ layer.weight: torch.bfloat16                │
│   shape: [output_size, input_size]          │
│   显存: output × input × 2 bytes            │
│                                             │
│ layer.weight_scale: 不存在                  │
└─────────────────────────────────────────────┘
                    │
                    │  ops.scaled_fp8_quant()
                    │
                    ▼
量化后:
┌─────────────────────────────────────────────┐
│ layer.weight: torch.float8_e4m3fn           │
│   shape: [input_size, output_size] (转置)   │
│   显存: output × input × 1 byte             │
│                                             │
│ layer.weight_scale: torch.float32           │
│   shape: [] (标量)                          │
│   显存: 4 bytes                             │
└─────────────────────────────────────────────┘

显存节省: ~50%
```

---

## 9. 阶段六：推理计算

### 9.1 generate 调用链

```
omni.generate(prompt)
    │
    ├──► engine.add_request()      # 添加请求到队列
    │
    └──► engine.try_get_output()   # 获取输出
            │
            └──► transformer.forward()
```

### 9.2 Transformer 前向传播

**文件**: `vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py`

```python
class Flux2KleinTransformer2DModel:
    def forward(self, hidden_states, timestep, encoder_hidden_states, ...):
        # 去噪循环
        for block in self.transformer_blocks:
            # ========== 量化关键：Linear 层计算 ==========
            hidden_states = block(
                hidden_states,
                timestep,
                encoder_hidden_states,
                ...
            )
        
        return hidden_states
```

### 9.3 Block 前向传播

```python
class Flux2KleinBlock:
    def forward(self, hidden_states, ...):
        # Self-Attention
        # ========== 量化关键：to_qkv 是 Linear 层 ==========
        q, k, v = self.attn.to_qkv(hidden_states).chunk(3, dim=-1)
        
        attn_output = attention(q, k, v)
        hidden_states = self.attn.to_out(attn_output)  # 量化
        
        # MLP
        # ========== 量化关键：ff 是 MLP 层 ==========
        hidden_states = self.ff(hidden_states)  # 量化
        
        return hidden_states
```

### 9.4 Linear 层前向传播

**文件**: `vllm/model_executor/layers/linear.py`

```python
class LinearBase:
    def forward(self, x):
        # ========== 量化核心：调用量化方法 ==========
        return self.quant_method.apply(self, x, self.bias)
```

### 9.5 Fp8LinearMethod.apply

**文件**: `vllm/model_executor/layers/quantization/fp8.py`

```python
class Fp8LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Fp8Config):
        # 选择最优 kernel
        if cutlass_fp8_supported():
            activation_key = kFp8DynamicTokenSym
        else:
            activation_key = kFp8DynamicTensorSym
        
        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=activation_key,
            weight_quant_key=kFp8StaticTensorSym,
            out_dtype=torch.bfloat16,
        )
    
    def apply(self, layer, x, bias=None):
        """执行 FP8 线性层计算
        
        y = (x @ W_fp8.T) * scale_x * scale_w + bias
        
        参数:
            layer: 包含 FP8 权重的层
                - layer.weight: FP8 权重 (转置后)
                - layer.weight_scale: 权重 scale
            x: 输入 (BF16)
            bias: 偏置
        
        返回:
            输出 (BF16)
        """
        return self.fp8_linear.apply_weights(layer, x, bias)
```

### 9.6 FP8 GEMM 计算流程

```
输入 x (BF16)              权重 W (FP8) + scale
      │                           │
      ▼                           │
┌─────────────────┐              │
│ 动态量化到 FP8   │              │
│                 │              │
│ scale_x =       │              │
│   max(|x|)/448  │              │
│                 │              │
│ x_fp8 =         │              │
│   quant(x,      │              │
│     scale_x)    │              │
└─────────────────┘              │
      │                          │
      ▼                          ▼
┌─────────────────────────────────────────────────────┐
│              FP8 GEMM (torch._scaled_mm)            │
│                                                     │
│  CUDA Kernel 执行:                                  │
│    output_bf16 = (x_fp8 @ W_fp8)                   │
│                  * scale_x * scale_w               │
│                                                     │
│  硬件加速:                                          │
│    - Ada/Hopper: 原生 FP8 Tensor Core              │
│    - Ampere: Marlin 模拟                           │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
              输出 (BF16)
```

---

## 10. FP8 量化核心实现

### 10.1 CUDA Kernel: scaled_fp8_quant

```python
# vllm/_custom_ops.py (简化示意)

def scaled_fp8_quant(
    weight: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 量化的 CUDA 实现
    
    数学:
        scale = max(|weight|) / 448.0
        weight_fp8 = round(weight / scale).clamp(-448, 447)
        weight_fp8 = weight_fp8.to(float8_e4m3fn)
    """
    if scale is None:
        # 并行计算最大绝对值
        scale = weight.abs().max() / 448.0
    
    # 并行量化
    quantized = (weight / scale).round().clamp(-448, 447)
    quantized = quantized.to(torch.float8_e4m3fn)
    
    return quantized, scale
```

### 10.2 Kernel 选择策略

```python
def init_fp8_linear_kernel(
    activation_quant_key: str,
    weight_quant_key: str,
    out_dtype: torch.dtype,
) -> LinearKernel:
    """根据硬件选择 kernel"""
    
    # 量化类型
    # - kFp8DynamicTokenSym: Per-token 动态量化（推荐）
    # - kFp8DynamicTensorSym: Per-tensor 动态量化
    # - kFp8StaticTensorSym: 静态量化
    
    if cutlass_fp8_supported():
        # Ada/Hopper: 原生 FP8
        return CutlassFp8LinearKernel(...)
    elif marlin_fp8_supported():
        # Ampere: Marlin 模拟
        return MarlinFP8ScaledMMLinearKernel(...)
    else:
        return TorchScaledMMLinearKernel(...)
```

---

## 11. 示例：FLUX.2-klein FP8 量化

### 11.1 完整代码

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# 初始化（触发所有阶段）
omni = Omni(
    model="black-forest-labs/FLUX.2-klein-4B",
    quantization="fp8",
)

# 推理
outputs = omni.generate(
    "A beautiful sunset over the ocean",
    OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=20,
    ),
)

# 保存
images = outputs[0].request_output.images
images[0].save("output.png")
```

### 11.2 量化层统计

| 层类型 | 数量 | 原始显存 | FP8 显存 |
|--------|------|----------|----------|
| to_qkv | 19 | 2.1GB | 1.05GB |
| to_out | 19 | 0.7GB | 0.35GB |
| ff.net.0 | 19 | 4.2GB | 2.1GB |
| ff.net.2 | 19 | 4.2GB | 2.1GB |
| **总计** | 76 | **11.2GB** | **5.6GB** |

### 11.3 关键文件路径

| 文件 | 作用 |
|------|------|
| `vllm_omni/entrypoints/omni.py` | 用户入口 |
| `vllm_omni/entrypoints/omni_base.py` | 引擎初始化 |
| `vllm_omni/engine/async_omni_engine.py` | 异步引擎 |
| `vllm_omni/engine/stage_init_utils.py` | 阶段初始化 |
| `vllm_omni/diffusion/data.py` | 配置构建 |
| `vllm_omni/quantization/factory.py` | 量化配置工厂 |
| `vllm_omni/diffusion/registry.py` | 模型注册 |
| `vllm_omni/diffusion/model_loader/diffusers_loader.py` | 权重加载 |
| `vllm_omni/diffusion/worker/diffusion_model_runner.py` | 模型运行器 |
| `vllm/model_executor/layers/quantization/fp8.py` | FP8 核心实现 |
| `vllm/model_executor/layers/linear.py` | Linear 层 |
