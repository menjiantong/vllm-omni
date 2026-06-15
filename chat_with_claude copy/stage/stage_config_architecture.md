# vLLM-Omni Stage Config 架构详解

本文档结合 MammothModa2 模型集成过程中遇到的三个配置问题，详细说明 vLLM-Omni 的 stage_config 工作原理、启动流程、相关组件及其交互。

---

## 目录

1. [概述](#1-概述)
2. [三个配置问题的根因分析](#2-三个配置问题的根因分析)
3. [Stage Config 架构](#3-stage-config-架构)
4. [启动流程](#4-启动流程)
5. [配置传播链](#5-配置传播链)
6. [多进程架构](#6-多进程架构)
7. [最佳实践](#7-最佳实践)

---

## 1. 概述

### 1.1 什么是 Stage Config？

vLLM-Omni 采用**多阶段流水线架构**来处理多模态生成任务。Stage Config 是定义每个阶段行为的配置文件，以 YAML 格式存储。

例如，MammothModa2 的 T2I（文本生成图像）流程包含两个阶段：

```
┌─────────────┐         ┌─────────────┐
│  Stage 0    │  latent │  Stage 1    │
│  AR Stage   │ ──────► │  DiT Stage  │
│  (理解+编码) │         │  (解码图像)  │
└─────────────┘         └─────────────┘
```

### 1.2 配置层级结构

```
Mammothmoda2Config (顶层配置)
├── llm_config: Mammothmoda2Qwen2_5_VLConfig (VL 多模态配置)
│   ├── text_config: Mammothmoda2Qwen2_5_VLTextConfig (文本模型配置)
│   │   ├── hidden_size: 8192
│   │   ├── num_hidden_layers: 80
│   │   ├── num_attention_heads: 64
│   │   └── ...
│   └── vision_config: Mammothmoda2Qwen2_5_VLVisionConfig
├── gen_vae_config: dict (VAE 配置)
├── gen_dit_config: dict (DiT 配置)
└── ...
```

---

## 2. 三个配置问题的根因分析

### 2.1 问题一：`'Mammothmoda2Config' object has no attribute 'llm_config'`

**现象：**
```python
AttributeError: 'Mammothmoda2Config' object has no attribute 'llm_config'.
Did you mean: 'sub_configs'?
```

**根因分析：**

```
PretrainedConfig.__init__()
    │
    ├── 内部调用 validate() 验证器
    │       │
    │       └── validate_token_ids()
    │               │
    │               └── self.get_text_config(decoder=True)
    │                       │
    │                       └── return self.llm_config  # 此时 llm_config 未赋值！
    │
    └── super().__init__() 返回后
            │
            └── self.llm_config = ...  # 太晚了
```

**原始代码（错误）：**
```python
class Mammothmoda2Config(PretrainedConfig):
    def __init__(self, *, llm_config: dict | None = None, ...):
        super().__init__(**kwargs)        # ← 触发验证
        self.llm_config = AutoConfig.for_model(**llm_config)  # ← 太晚赋值
```

**修复方案：**
```python
class Mammothmoda2Config(PretrainedConfig):
    def __init__(self, *, llm_config: dict | None = None, ...):
        # 先设置所有属性
        self.llm_config = AutoConfig.for_model(**llm_config) if llm_config else None
        self.gen_vae_config = gen_vae_config
        # ... 其他属性

        # 最后调用父类初始化
        super().__init__(**kwargs)
```

**关键点：** Python 数据类/PretrainedConfig 的验证机制会在 `__init__` 内部触发，必须确保验证所需的属性在 `super().__init__()` 之前已设置。

---

### 2.2 问题二：`text_config does not have 'num_attention_heads' attribute`

**现象：**
```python
ValidationError: The text_config extracted from the model config
does not have `num_attention_heads` attribute.
```

**根因分析：**

vLLM 期望 `get_text_config()` 返回**真正的文本模型配置**，但原代码返回了 VL 配置层：

```
期望的返回值：
Mammothmoda2Qwen2_5_VLTextConfig  # 有 num_attention_heads

实际返回值：
Mammothmoda2Qwen2_5_VLConfig      # 没有 num_attention_heads（它是 VL 配置层）
```

**原始代码（错误）：**
```python
def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
    return self.llm_config  # 返回 VL 配置，不是文本配置
```

**修复方案：**
```python
def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
    if self.llm_config is None:
        return None
    # 返回嵌套的 text_config
    return self.llm_config.text_config
```

**关键点：** 三层嵌套配置结构中，`get_text_config()` 必须穿透到最内层的文本配置，而非中间的 VL 配置层。

---

### 2.3 问题三：`Failed to infer llm hidden_size`

**现象：**
```python
ValueError: Failed to infer llm hidden_size from
Mammothmoda2Config.llm_config.hidden_size
```

**根因分析：**

DiT Pipeline 尝试从 `llm_config` 获取 `hidden_size`，但该属性在嵌套的 `text_config` 中：

```python
# 错误的访问路径
config.llm_config.hidden_size      # Mammothmoda2Qwen2_5_VLConfig 没有 hidden_size

# 正确的访问路径
config.llm_config.text_config.hidden_size  # 8192
```

**原始代码（错误）：**
```python
llm_hidden_size = int(getattr(self.config.llm_config, "hidden_size", 0) or 0)
```

**修复方案：**
```python
# llm_config is a Mammothmoda2Qwen2_5_VLConfig which has nested text_config
llm_hidden_size = int(
    getattr(
        getattr(self.config.llm_config, "text_config", None),
        "hidden_size",
        0,
    )
    or 0
)
```

**关键点：** 访问三层嵌套配置的属性时，必须清楚每一层的职责和包含的属性。

---

## 3. Stage Config 架构

### 3.1 配置文件格式

**位置：** `vllm_omni/model_executor/stage_configs/*.yaml`

**示例（mammoth_moda2.yaml）：**
```yaml
stage_args:
  - stage_id: 0
    runtime:
      devices: "0"
    engine_args:
      model_stage: ar
      max_num_seqs: 100
      model_arch: MammothModa2ForConditionalGeneration
      worker_cls: vllm_omni.worker.gpu_ar_worker.GPUARWorker
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      max_model_len: 8192
      gpu_memory_utilization: 0.5
      enforce_eager: true
      trust_remote_code: true
      engine_output_type: latent
      enable_prefix_caching: false
    final_output: false

  - stage_id: 1
    runtime:
      devices: "0"
    engine_args:
      model_stage: dit
      max_num_seqs: 1
      model_arch: MammothModa2ForConditionalGeneration
      worker_cls: vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      gpu_memory_utilization: 0.3
      enforce_eager: true
      trust_remote_code: true
      engine_output_type: image
      enable_prefix_caching: false
    engine_input_source: [0]
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.mammoth_moda2.ar2dit
    final_output: true
    final_output_type: image
```

### 3.2 关键配置字段说明

| 字段 | 说明 | 示例 |
|------|------|------|
| `stage_id` | 阶段唯一标识 | 0, 1 |
| `runtime.devices` | 使用的 GPU 设备 | "0", "0,1" |
| `engine_args.model_stage` | 模型阶段类型 | ar, dit, vae |
| `engine_args.model_arch` | 模型架构类名 | MammothModa2ForConditionalGeneration |
| `engine_args.worker_cls` | Worker 类全路径 | GPUARWorker |
| `engine_args.scheduler_cls` | 调度器类全路径 | OmniARScheduler |
| `engine_args.engine_output_type` | 输出类型 | latent, image |
| `engine_input_source` | 输入来源阶段 | [0] 表示来自 stage 0 |
| `final_output` | 是否为最终输出阶段 | true, false |
| `custom_process_input_func` | 自定义输入处理函数 | ar2dit |

### 3.3 配置加载路径

```
┌─────────────────────────────────────────────────────────────────┐
│                      配置加载入口                                │
│  load_and_resolve_stage_configs() in entrypoints/utils.py       │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌─────────────────────┐       ┌─────────────────────┐
    │  新路径：Pipeline    │       │  旧路径：YAML       │
    │  Registry + Deploy  │       │  stage_args 格式    │
    └─────────────────────┘       └─────────────────────┘
              │                               │
              ▼                               ▼
    ┌─────────────────────┐       ┌─────────────────────┐
    │ StageConfigFactory  │       │ load_stage_configs  │
    │ .create_from_model()│       │ _from_yaml()        │
    └─────────────────────┘       └─────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  list[StageConfig]  │
                    └─────────────────────┘
```

---

## 4. 启动流程

### 4.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           用户调用                                       │
│  omni = Omni(model="...", stage_configs_path="...")                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        AsyncOmniEngine.__init__()                        │
│  1. _resolve_stage_configs() → 加载并解析 stage configs                  │
│  2. 创建 Janus 队列用于线程间通信                                         │
│  3. 启动 Orchestrator 后台线程                                           │
│  4. 等待 Orchestrator 初始化完成                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Orchestrator 线程 (_bootstrap_orchestrator)           │
│  创建独立的 asyncio 事件循环                                              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    _initialize_stages()                             │ │
│  │  1. compute_replica_layout() → 计算副本布局                         │ │
│  │  2. _build_logical_stage_init_plans() → 构建初始化计划              │ │
│  │  3. _initialize_stage_replicas() → 初始化所有副本                   │ │
│  │  4. _assemble_stage_pools() → 组装 StagePools                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      Orchestrator.run()                             │ │
│  │  主循环：处理请求、调度阶段、收集输出                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Stage 初始化详细流程

```
_initialize_stages()
        │
        ├── Phase 1: compute_replica_layout()
        │       └── 根据设备配置计算每个阶段的副本数量和分布
        │
        ├── Phase 2: _build_logical_stage_init_plans()
        │       │
        │       ├── for each stage_config:
        │       │       │
        │       │       ├── extract_stage_metadata() → StageMetadata
        │       │       │
        │       │       ├── build_engine_args_dict() → dict
        │       │       │
        │       │       ├── build_vllm_config() → VllmConfig
        │       │       │       │
        │       │       │       ├── OmniEngineArgs(**engine_args_dict)
        │       │       │       │
        │       │       │       └── create_engine_config()
        │       │       │               │
        │       │       │               └── create_model_config()
        │       │       │                       │
        │       │       │                       └── AutoConfig.from_pretrained()
        │       │       │                               │
        │       │       │                               └── 返回 Mammothmoda2Config
        │       │       │
        │       │       └── 创建 ReplicaInitPlan 列表
        │       │
        │       └── 返回 list[LogicalStageInitPlan]
        │
        ├── Phase 3: _initialize_stage_replicas()
        │       │
        │       └── for each replica_plan:
        │               │
        │               ├── _initialize_llm_replica() 或 _initialize_diffusion_replica()
        │               │       │
        │               │       ├── spawn_stage_core() → 启动子进程
        │               │       │       │
        │               │       │       └── StageEngineCoreProc.run_stage_core()
        │               │       │               │
        │               │       │               ├── 创建 EngineCore
        │               │       │               │
        │               │       │               └── 运行 busy loop
        │               │       │
        │               │       ├── complete_stage_handshake() → HELLO/INIT/READY 握手
        │               │       │
        │               │       └── 创建 StageEngineCoreClient
        │               │
        │               └── 返回 StageEngineCoreClient
        │
        └── Phase 4: _assemble_stage_pools()
                └── 将 clients 组装成 StagePools 供 Orchestrator 调度
```

### 4.3 握手协议（Handshake）

子进程启动后，需要与主进程进行三次握手：

```
主进程                                    子进程 (StageEngineCoreProc)
   │                                           │
   │◄──────────── HELLO ───────────────────────│ 子进程就绪
   │                                           │
   │───────────── INIT ───────────────────────►│ 发送初始化信息
   │        (vllm_config, addresses)           │
   │                                           │
   │◄──────────── READY ──────────────────────│ 初始化完成
   │                                           │
   │                                           │ 开始处理请求
```

---

## 5. 配置传播链

### 5.1 从 CLI 到模型实例

```
CLI 参数 / kwargs
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      _resolve_stage_configs()                            │
│  加载 YAML，合并 CLI 覆盖项                                               │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         StageConfig                                      │
│  每个阶段一份，包含 engine_args、runtime、metadata                        │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      build_engine_args_dict()                            │
│  将 StageConfig 转换为 EngineArgs 兼容的字典                              │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OmniEngineArgs                                   │
│  继承自 vLLM EngineArgs，添加 omni 特定参数                               │
│                                                                          │
│  create_model_config()                                                   │
│      │                                                                   │
│      └── super().create_model_config()                                   │
│              │                                                           │
│              └── ModelConfig.__post_init__()                             │
│                      │                                                   │
│                      └── get_config() → AutoConfig.from_pretrained()     │
│                              │                                           │
│                              └── 返回 Mammothmoda2Config 实例             │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          VllmConfig                                      │
│  包含 model_config、cache_config、quant_config 等                         │
│                                                                          │
│  vllm_config.model_config.hf_config → Mammothmoda2Config                 │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    模型初始化 (MammothModa2ForConditionalGeneration)     │
│                                                                          │
│  cfg = vllm_config.model_config.hf_config  # Mammothmoda2Config          │
│                                                                          │
│  if model_stage == "ar":                                                 │
│      hf_config = cfg.llm_config  # 传递 VL 配置给 AR 子模型              │
│  elif model_stage == "dit":                                              │
│      hf_config = cfg  # DiT 使用顶层配置，内部读取 gen_dit_config        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 配置层级与访问规则

```
顶层配置：Mammothmoda2Config
    │
    │  访问规则：顶层配置提供聚合视图
    │  - get_text_config() → 返回 llm_config.text_config
    │  - vision_config (property) → 返回 llm_config.vision_config
    │  - image_token_id (property) → 返回 llm_config.image_token_id
    │
    ├── llm_config: Mammothmoda2Qwen2_5_VLConfig (VL 配置层)
    │       │
    │       │  职责：管理视觉-语言多模态组件
    │       │  - 包含 text_config 和 vision_config
    │       │  - 提供 vision 相关的 token ID
    │       │
    │       ├── text_config: Mammothmoda2Qwen2_5_VLTextConfig
    │       │       │
    │       │       │  职责：纯文本模型配置
    │       │       │  - hidden_size, num_hidden_layers, num_attention_heads
    │       │       │  - vocab_size, intermediate_size
    │       │       │  - extra_gen_vocab, gen_vocab_size
    │       │       │
    │       │       └── 这是 vLLM 期望从 get_text_config() 获得的对象
    │       │
    │       └── vision_config: Mammothmoda2Qwen2_5_VLVisionConfig
    │               │
    │               └── 视觉编码器配置
    │
    ├── gen_vae_config: dict (VAE 配置，diffusers 格式)
    │
    └── gen_dit_config: dict (DiT 配置，diffusers 格式)
```

### 5.3 属性代理模式

为兼容 vLLM 的多模态代码路径，顶层配置使用 property 代理访问嵌套属性：

```python
class Mammothmoda2Config(PretrainedConfig):
    @property
    def vision_config(self):
        return self._require_llm_config().vision_config

    @property
    def image_token_id(self) -> int:
        return int(self._require_llm_config().image_token_id)

    @property
    def video_token_id(self) -> int:
        return int(self._require_llm_config().video_token_id)
```

这样 vLLM 的 `mrope.py` 等模块可以直接访问 `hf_config.vision_config`，无需感知三层嵌套结构。

---

## 6. 多进程架构

### 6.1 进程结构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              主进程                                      │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     AsyncOmniEngine                                 │ │
│  │  - request_queue (Janus Queue)                                      │ │
│  │  - output_queue (Janus Queue)                                       │ │
│  │  - stage_pools: dict[int, StagePool]                                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                   Orchestrator Thread (daemon)                      │ │
│  │  - asyncio event loop                                               │ │
│  │  - Orchestrator 实例                                                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │ StageEngine   │ │ StageEngine   │ │ Diffusion     │
        │ CoreProc #0   │ │ CoreProc #1   │ │ Worker #0     │
        │ (LLM Stage)   │ │ (LLM Stage)   │ │ (Diff Stage)  │
        └───────────────┘ └───────────────┘ └───────────────┘
              GPU 0             GPU 1             GPU 0
```

### 6.2 通信机制

```
主进程                                    子进程
   │                                         │
   │  ┌─────────────────────────────────────┐│
   │  │         ZMQ Sockets                 ││
   │  │  - input_socket (REQ/REP)           ││
   │  │  - output_socket (REQ/REP)          ││
   │  │  - handshake_socket (REQ/REP)       ││
   │  └─────────────────────────────────────┘│
   │                                         │
   │  通过 ZMQ 发送：                         │
   │  - 请求 (SchedulerOutput)               │
   │  - 响应 (EngineCoreOutput)              │
   │                                         │
```

### 6.3 设备锁机制

为避免多个阶段同时初始化时的 GPU 竞争，使用文件锁：

```python
def acquire_device_locks(stage_id, engine_args_dict, stage_init_timeout):
    for device_id in devices_to_lock:
        lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
```

这确保同一 GPU 上的多个阶段按顺序初始化。

---

## 7. 最佳实践

### 7.1 配置类设计原则

1. **初始化顺序**：属性赋值在 `super().__init__()` 之前完成
2. **get_text_config()**：必须返回包含 `num_attention_heads` 等属性的实际文本配置
3. **属性代理**：使用 `@property` 代理嵌套属性，保持接口兼容

### 7.2 多层嵌套配置的处理

```python
# 推荐：明确的访问路径
hidden_size = config.llm_config.text_config.hidden_size

# 或使用安全的 getattr 链
hidden_size = getattr(
    getattr(config.llm_config, "text_config", None),
    "hidden_size",
    default_value
)
```

### 7.3 Stage Config 编写建议

1. **显式指定 model_stage**：每个阶段的 `model_stage` 必须与模型代码中的分支匹配
2. **engine_input_source**：确保阶段间的依赖关系正确
3. **gpu_memory_utilization**：根据模型大小合理分配，总和不超过 1.0（单 GPU 场景）
4. **devices**：多 GPU 场景下合理分配设备

### 7.4 调试技巧

1. **查看配置解析结果**：
   ```python
   from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs
   configs = load_and_resolve_stage_configs(model="...", stage_configs_path="...")
   for cfg in configs:
       print(cfg)
   ```

2. **检查 hf_config 结构**：
   ```python
   from transformers import AutoConfig
   config = AutoConfig.from_pretrained("bytedance-research/MammothModa2-Preview", trust_remote_code=True)
   print(type(config))
   print(hasattr(config, "llm_config"))
   print(config.get_text_config())
   ```

3. **子进程日志**：StageEngineCoreProc 的错误日志会输出到 `(StageEngineCoreProc pid=XXXX)` 前缀的行中。

---

## 附录：关键文件索引

| 组件 | 文件路径 |
|------|----------|
| AsyncOmniEngine | `vllm_omni/engine/async_omni_engine.py` |
| Stage 初始化工具 | `vllm_omni/engine/stage_init_utils.py` |
| Stage 子进程 | `vllm_omni/engine/stage_engine_core_proc.py` |
| Stage Config 系统 | `vllm_omni/config/stage_config.py` |
| Pipeline 注册表 | `vllm_omni/config/pipeline_registry.py` |
| 配置加载入口 | `vllm_omni/entrypoints/utils.py` |
| OmniBase 入口 | `vllm_omni/entrypoints/omni_base.py` |
| OmniEngineArgs | `vllm_omni/engine/arg_utils.py` |
| MammothModa2 配置 | `vllm_omni/transformers_utils/configs/mammoth_moda2.py` |
| MammothModa2 模型 | `vllm_omni/model_executor/models/mammoth_moda2/mammoth_moda2.py` |
| MammothModa2 Stage YAML | `vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml` |
| MammothModa2 DiT Pipeline | `vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py` |
