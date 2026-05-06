# DiffusionModelRunner 详解

## 一、概述

`DiffusionModelRunner` 是 vLLM-Omni 中负责扩散模型执行的核心组件。它遵循 **AR (Actor-Runner) 模式**，其中 Runner 负责所有与模型相关的操作，而 Worker 只负责基础设施管理。

### 类继承关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              类继承关系                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   OmniConnectorModelRunnerMixin (父类 Mixin)                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  - 跨阶段数据传输 (connector)                                        │  │
│   │  - KV Cache 传输管理                                                 │  │
│   │  - 流式 chunk 收发                                                   │  │
│   │  - 后台 I/O 线程                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                   ▲                                        │
│                                   │ 继承                                    │
│                                   │                                        │
│   DiffusionModelRunner (核心类)                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  - 扩散模型加载                                                      │  │
│   │  - 模型编译 (torch.compile)                                          │  │
│   │  - Cache-DiT 加速后端                                                │  │
│   │  - 模型推理执行 (execute_model / execute_stepwise)                   │  │
│   │  - CPU/Layer-wise Offload                                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、生命周期

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DiffusionModelRunner 生命周期                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Phase 1: 初始化 (__init__)                                                │
│   ══════════════════════════                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  输入: vllm_config, od_config, device                               │  │
│   │                                                                      │  │
│   │  初始化内容:                                                         │  │
│   │  ├── self.pipeline = None          # 扩散模型 Pipeline              │  │
│   │  ├── self.cache_backend = None     # Cache-DiT 加速后端             │  │
│   │  ├── self.offload_backend = None   # CPU Offload 后端               │  │
│   │  ├── self.state_cache = {}         # Step-wise 请求状态缓存         │  │
│   │  └── self.kv_transfer_manager      # KV Cache 跨阶段传输管理        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Phase 2: 模型加载 (load_model)                                            │
│   ═════════════════════════════                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  1. 使用 DiffusersPipelineLoader 加载模型                           │  │
│   │     └── 支持 HuggingFace Diffusers 格式                             │  │
│   │                                                                      │  │
│   │  2. 应用 CPU Offload (如果启用)                                      │  │
│   │     ├── enable_cpu_offload → 移动模块到 CPU                         │  │
│   │     └── enable_layerwise_offload → 逐层 Offload                     │  │
│   │                                                                      │  │
│   │  3. 应用 torch.compile (如果未启用 eager mode)                       │  │
│   │     └── regionally_compile(transformer)                             │  │
│   │                                                                      │  │
│   │  4. 启用 Cache-DiT 加速 (如果配置)                                   │  │
│   │     └── cache_backend.enable(pipeline)                              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Phase 3: 推理执行 (execute_model / execute_stepwise)                      │
│   ══════════════════════════════════════════════════                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  两种执行模式:                                                       │  │
│   │                                                                      │  │
│   │  ├── execute_model()      # 一次性完整推理                          │  │
│   │  │   └── pipeline.forward(req) → DiffusionOutput                   │  │
│   │  │                                                                   │  │
│   │  └── execute_stepwise()   # 逐步推理 (支持调度器控制)               │  │
│   │      ├── prepare_encode(state)                                      │  │
│   │      ├── denoise_step(state)                                        │  │
│   │      ├── step_scheduler(state)                                      │  │
│   │      └── post_decode(state) → RunnerOutput                          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Phase 4: 清理 (shutdown)                                                  │
│   ════════════════════════                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  - 释放 pipeline 资源                                                │  │
│   │  - 清理 cache_backend                                                │  │
│   │  - 清理 state_cache                                                  │  │
│   │  - 调用 shutdown_omni_connectors() (来自父类)                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、核心机制

### 3.1 模型加载机制

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            模型加载流程                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   load_model(load_format, custom_pipeline_name)                             │
│   │                                                                         │
│   ├── 1. 确定加载设备                                                       │
│   │   ├── cpu_offload 或 layerwise_offload → load_device = "cpu"           │
│   │   └── 否则 → load_device = self.device                                 │
│   │                                                                         │
│   ├── 2. 创建内存池上下文 (支持 sleep mode)                                  │
│   │   └── memory_pool_context_fn(tag="weights")                            │
│   │                                                                         │
│   ├── 3. 使用 DiffusersPipelineLoader 加载                                  │
│   │   ┌───────────────────────────────────────────────────────────┐       │
│   │   │  model_loader = DiffusersPipelineLoader(load_config)      │       │
│   │   │  pipeline = model_loader.load_model(                      │       │
│   │   │      od_config=od_config,                                 │       │
│   │   │      load_device=load_device,                             │       │
│   │   │      load_format=load_format,                             │       │
│   │   │      custom_pipeline_name=custom_pipeline_name,           │       │
│   │   │  )                                                        │       │
│   │   └───────────────────────────────────────────────────────────┘       │
│   │                                                                         │
│   ├── 4. 验证 step_execution 支持                                           │
│   │   └── 如果 step_execution=True，检查 pipeline 是否实现必要接口         │
│   │                                                                         │
│   ├── 5. 应用 CPU Offload                                                   │
│   │   ┌───────────────────────────────────────────────────────────┐       │
│   │   │  offload_backend = get_offload_backend(od_config)         │       │
│   │   │  offload_backend.enable(pipeline)                         │       │
│   │   │                                                           │       │
│   │   │  支持类型:                                                │       │
│   │   │  ├── CPU Offload: 模块在 CPU，计算时移动到 GPU            │       │
│   │   │  └── Layerwise Offload: 逐层加载到 GPU                    │       │
│   │   └───────────────────────────────────────────────────────────┘       │
│   │                                                                         │
│   ├── 6. 应用 torch.compile                                                 │
│   │   ┌───────────────────────────────────────────────────────────┐       │
│   │   │  if not enforce_eager:                                    │       │
│   │   │      _compile_transformer("transformer")                  │       │
│   │   │      _compile_transformer("transformer_2")  # 如有        │       │
│   │   │                                                           │       │
│   │   │  使用 regionally_compile 进行动态编译                      │       │
│   │   └───────────────────────────────────────────────────────────┘       │
│   │                                                                         │
│   └── 7. 启用 Cache-DiT 加速                                                │
│       ┌───────────────────────────────────────────────────────────┐       │
│       │  cache_backend = get_cache_backend(cache_backend_name)   │       │
│       │  cache_backend.enable(pipeline)                           │       │
│       │                                                           │       │
│       │  Cache-DiT: 缓存 DiT 中间激活，减少重复计算                │       │
│       └───────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 两种执行模式对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          执行模式对比                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    execute_model (一次性模式)                        │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │                                                                      │  │
│   │   输入: OmniDiffusionRequest (完整请求)                             │  │
│   │   输出: DiffusionOutput (完整输出)                                  │  │
│   │                                                                      │  │
│   │   流程:                                                              │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │  1. 接收 KV Cache (跨阶段传输)                               │   │  │
│   │   │  2. 初始化随机数生成器                                       │   │  │
│   │   │  3. 刷新 Cache Backend (Cache-DiT)                          │   │  │
│   │   │  4. pipeline.forward(req)                                    │   │  │
│   │   │     └── 内部完成所有 diffusion steps                        │   │  │
│   │   │  5. 记录峰值内存                                             │   │  │
│   │   │  6. 返回输出                                                 │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                                                                      │  │
│   │   特点:                                                              │  │
│   │   - 简单直接，适合单次请求                                          │  │
│   │   - 所有 step 在内部完成                                            │  │
│   │   - 不支持精细的调度控制                                             │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                  execute_stepwise (逐步模式)                         │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │                                                                      │  │
│   │   输入: DiffusionSchedulerOutput (调度器输出)                       │  │
│   │   输出: RunnerOutput (单步结果)                                     │  │
│   │                                                                      │  │
│   │   流程:                                                              │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │  每次调用执行一步:                                           │   │  │
│   │   │                                                               │   │  │
│   │   │  1. _update_states(scheduler_output)                         │   │  │
│   │   │     └── 管理请求状态缓存 (self.state_cache)                  │   │  │
│   │   │                                                               │   │  │
│   │   │  2. if is_new_request:                                       │   │  │
│   │   │        pipeline.prepare_encode(state)  # 编码 prompt        │   │  │
│   │   │                                                               │   │  │
│   │   │  3. noise_pred = pipeline.denoise_step(state)  # 一次去噪   │   │  │
│   │   │                                                               │   │  │
│   │   │  4. pipeline.step_scheduler(state, noise_pred)  # 更新隐变量│   │  │
│   │   │                                                               │   │  │
│   │   │  5. if finished:                                             │   │  │
│   │   │        result = pipeline.post_decode(state)  # 解码输出     │   │  │
│   │   │                                                               │   │  │
│   │   │  6. _update_states_after(state, finished)                    │   │  │
│   │   │     └── 清理完成请求的状态                                    │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                                                                      │  │
│   │   特点:                                                              │  │
│   │   - 支持调度器精细控制                                               │  │
│   │   - 可中断、可抢占                                                   │  │
│   │   - 支持多请求交错执行                                               │  │
│   │   - 需要 pipeline 实现特定接口                                      │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Step-wise 接口要求

```python
# Pipeline 必须实现以下接口才能支持 step_execution:

class StepwisePipeline:
    def prepare_encode(self, state: DiffusionRequestState) -> None:
        """初始化请求状态，编码 prompt"""
        pass

    def denoise_step(self, state: DiffusionRequestState) -> torch.Tensor | None:
        """执行一次去噪步骤，返回噪声预测"""
        pass

    def step_scheduler(self, state: DiffusionRequestState, noise_pred: torch.Tensor) -> None:
        """使用调度器更新隐变量"""
        pass

    def post_decode(self, state: DiffusionRequestState) -> DiffusionOutput:
        """解码最终隐变量，返回输出"""
        pass
```

### 3.4 Cache-DiT 加速机制

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Cache-DiT 加速原理                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   问题: DiT (Diffusion Transformer) 的重复计算                              │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  标准 Diffusion 推理:                                             │    │
│   │                                                                    │    │
│   │  Step 0: [Prompt] → [Noise] → DiT(全部层) → [Denoised]           │    │
│   │  Step 1: [Denoised] → DiT(全部层) → [Denoised]                   │    │
│   │  Step 2: [Denoised] → DiT(全部层) → [Denoised]                   │    │
│   │  ...                                                               │    │
│   │  Step N: [Denoised] → DiT(全部层) → [Output]                     │    │
│   │                                                                    │    │
│   │  每步都要重新计算整个 DiT，即使输入变化很小                        │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   解决方案: Cache-DiT                                                       │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  Cache-DiT 思路:                                                  │    │
│   │                                                                    │    │
│   │  Step 0: [Prompt] → [Noise] → DiT → 缓存中间激活                  │    │
│   │  Step 1: 复用缓存的激活 + 增量更新 → [Denoised]                   │    │
│   │  Step 2: 复用缓存的激活 + 增量更新 → [Denoised]                   │    │
│   │  ...                                                               │    │
│   │                                                                    │    │
│   │  关键洞察:                                                         │    │
│   │  - Diffusion steps 之间的隐变量变化较小                           │    │
│   │  - DiT 中间的 attention 输出变化也较小                             │    │
│   │  - 可以缓存并复用，只计算增量                                      │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   代码流程:                                                                 │
│   ─────────                                                                 │
│   cache_backend = get_cache_backend("cache_dit", cache_config)             │
│   cache_backend.enable(pipeline)  # 注入缓存逻辑到 pipeline                │
│                                                                             │
│   # 在 execute_model 中                                                    │
│   cache_backend.refresh(pipeline, num_inference_steps)                     │
│   # 根据步数刷新缓存策略                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、父类 OmniConnectorModelRunnerMixin 详解

### 4.1 核心职责

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 OmniConnectorModelRunnerMixin 核心职责                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        数据传输模式                                  │  │
│   │                                                                      │  │
│   │   1. full_payload_mode (完整负载模式)                               │  │
│   │      ├── recv_full_payload_inputs()  # 接收完整输入                 │  │
│   │      ├── send_full_payload_outputs() # 发送完整输出                 │  │
│   │      └── 用于: 跨阶段传递完整数据                                   │  │
│   │                                                                      │  │
│   │   2. async_chunk (异步流式模式)                                      │  │
│   │      ├── recv_chunk()  # 接收数据块                                 │  │
│   │      ├── send_chunk()  # 发送数据块                                 │  │
│   │      └── 用于: Thinker→Talker 流式传输                              │  │
│   │                                                                      │  │
│   │   3. KV Cache 传输                                                   │  │
│   │      ├── send_kv_cache()  # 发送 KV Cache                          │  │
│   │      ├── recv_kv_cache()  # 接收 KV Cache                          │  │
│   │      └── 用于: 跨阶段 KV Cache 复用                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        后台 I/O 线程                                 │  │
│   │                                                                      │  │
│   │   _recv_loop() ──────────────────────────────────────────────────┐  │  │
│   │   │  while not stopped:                                          │  │  │
│   │   │      for req_id in pending_load_reqs:                        │  │  │
│   │   │          connector.get(key)  # 从 connector 拉取数据         │  │  │
│   │   │          存入 _local_stage_payload_cache                     │  │  │
│   │   └──────────────────────────────────────────────────────────────┘  │  │
│   │                                                                      │  │
│   │   _save_loop() ──────────────────────────────────────────────────┐  │  │
│   │   │  while not stopped:                                          │  │  │
│   │   │      task = pending_save_reqs.popleft()                      │  │  │
│   │   │      connector.put(key, data)  # 发送到 connector            │  │  │
│   │   └──────────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        本地缓存管理                                  │  │
│   │                                                                      │  │
│   │   _local_stage_payload_cache: dict[str, dict]                      │  │
│   │   └── 存储从 connector 接收的数据                                   │  │
│   │                                                                      │  │
│   │   _local_request_metadata: dict[str, dict]                         │  │
│   │   └── 存储调度元数据 (如 prompt_len, left_context_size)            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 关键数据结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        关键数据结构                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   # 请求跟踪                                                                │
│   _pending_load_reqs: dict[str, Request]     # 等待接收的请求               │
│   _finished_load_reqs: set[str]              # 已完成接收的请求             │
│   _pending_save_reqs: dict[str, deque]       # 等待发送的任务队列           │
│                                                                             │
│   # Chunk 索引                                                              │
│   _put_req_chunk: dict[str, int]             # 发送 chunk 索引              │
│   _get_req_chunk: dict[str, int]             # 接收 chunk 索引              │
│                                                                             │
│   # 状态标志                                                                │
│   _chunk_ready_req_ids: set[str]             # Chunk 已就绪                 │
│   _chunk_finished_req_ids: set[str]          # Chunk 流已完成               │
│   _stage_recv_req_ids: set[str]              # 已接收数据的请求             │
│                                                                             │
│   # 本地缓存                                                                │
│   _local_stage_payload_cache: dict[str, dict]  # 负载缓存                  │
│   _local_request_metadata: dict[str, dict]     # 元数据缓存                 │
│                                                                             │
│   # KV Cache 传输状态                                                       │
│   _kv_pending_transfers: dict[str, dict]     # 待传输的 KV Cache           │
│   _kv_active_transfers: set[str]             # 正在传输中                   │
│   _kv_completed_transfers: set[str]          # 已完成传输                   │
│                                                                             │
│   # 发送端累积缓冲 (async_chunk)                                            │
│   _send_side_request_payload: dict[str, dict]  # 累积发送数据               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 数据传输流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     full_payload_mode 传输流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Stage 0 (Diffusion)                    Stage 1 (Next Stage)               │
│   ──────────────────                     ─────────────────────              │
│                                                                             │
│   execute_model()                                                              │
│       │                                                                        │
│       ▼                                                                        │
│   output = pipeline.forward(req)                                          │
│       │                                                                        │
│       ▼                                                                        │
│   accumulate_full_payload_output(req_id, output)                          │
│       │  # 累积输出 (跨多个 step)                                          │
│       ▼                                                                        │
│   flush_full_payload_outputs(finished_req_ids)                            │
│       │  # 请求完成时发送                                                   │
│       ▼                                                                        │
│   send_full_payload_outputs()                                                 │
│       │                                                                        │
│       │  connector.put(from_stage=0, to_stage=1, key, data)                │
│       │                                                                        │
│       └───────────────────────────────────────────────────────────────┐   │
│                                                                           │   │
│                                                                           ▼   │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │  Connector (如 Redis, Nccl, 等)                                     │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           ▲   │
│                                                                           │   │
│       ┌───────────────────────────────────────────────────────────────┘   │
│       │                                                                        │
│       ▼                                                                        │
│   _recv_loop() (后台线程)                                                     │
│       │  connector.get(from_stage=0, to_stage=1, key)                     │
│       ▼                                                                        │
│   存入 _local_stage_payload_cache                                          │
│       │                                                                        │
│       ▼                                                                        │
│   recv_full_payload_inputs()                                                 │
│       │  # 主线程检查缓存                                                   │
│       ▼                                                                        │
│   返回 {req_id: payload}                                                    │
│       │                                                                        │
│       ▼                                                                        │
│   使用 payload 作为下一阶段输入                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 async_chunk 流式传输

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      async_chunk 流式传输流程                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Thinker (Stage 0)                       Talker (Stage 1)                  │
│   ─────────────────                       ─────────────────                 │
│                                                                             │
│   每个 decode step:                                                            │
│       │                                                                        │
│       ▼                                                                        │
│   output = model.forward()                                                │
│       │                                                                        │
│       ▼                                                                        │
│   send_chunk(request, pooling_output)                                     │
│       │                                                                        │
│       │  ┌──────────────────────────────────────────────────────┐         │
│       │  │ payload = custom_process_func(pooling_output)        │         │
│       │  │ key = f"{req_id}_{stage}_{chunk_id}"                 │         │
│       │  │ connector.put(from_stage, to_stage, key, payload)    │         │
│       │  └──────────────────────────────────────────────────────┘         │
│       │                                                                        │
│       └───────────────────────────────────────────────────────────────┐   │
│                                                                           │   │
│                                                                           ▼   │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │  Connector 存储多个 chunks                                          │ │
│   │  key_0, key_1, key_2, ... key_N (finish sentinel)                  │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           ▲   │
│                                                                           │   │
│       ┌───────────────────────────────────────────────────────────────┘   │
│       │                                                                        │
│       ▼                                                                        │
│   _recv_loop() (后台线程)                                                     │
│       │  while True:                                                         │
│       │      for req_id in pending_load_reqs:                               │
│       │          result = connector.get(key)                                │
│       │          _accumulate_payload(req_id, result)  # 累积               │
│       │          if finished: break                                         │
│       ▼                                                                        │
│   recv_chunk()                                                              │
│       │  # 返回已就绪的 chunks                                              │
│       ▼                                                                        │
│   返回 {req_id: accumulated_payload}                                        │
│                                                                             │
│   特点:                                                                      │
│   ─────                                                                      │
│   - 异步: 发送和接收在不同线程                                               │
│   - 流式: 数据分块传输，不等待完整输出                                       │
│   - 累积: 接收端合并多个 chunk                                               │
│   - 终止: finish sentinel 标记流结束                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.5 KV Cache 传输

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        KV Cache 传输机制                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   用途: 在多阶段模型间复用 KV Cache，避免重复计算                            │
│                                                                             │
│   发送端 (Stage 0):                                                         │
│   ──────────────────                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  # 标记需要传输 KV Cache 的请求                                     │  │
│   │  mark_kv_transfer(req_id, seq_len, block_ids)                       │  │
│   │                                                                      │  │
│   │  # 提取并发送 KV Cache                                               │  │
│   │  send_kv_cache(finished_reqs, kv_caches, block_size, cache_dtype)  │  │
│   │      │                                                               │  │
│   │      └── kv_transfer_manager.handle_finished_requests_kv_transfer() │  │
│   │          │                                                           │  │
│   │          └── connector.put(key, kv_cache_data)                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   接收端 (Stage 1):                                                         │
│   ──────────────────                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  # 接收 KV Cache                                                     │  │
│   │  recv_kv_cache(request_id, target_device)                           │  │
│   │      │                                                               │  │
│   │      └── kv_transfer_manager.receive_kv_cache_for_request()         │  │
│   │          │                                                           │  │
│   │          └── connector.get(key)                                      │  │
│   │                                                                      │  │
│   │  # 应用到请求                                                        │  │
│   │  kv_transfer_manager.apply_kv_cache_to_request(req, data)           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   异构 TP 支持:                                                              │
│   ─────────────                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  当 Stage 0 和 Stage 1 的 TP 度不同时:                              │  │
│   │                                                                      │  │
│   │  from_tp > to_tp:                                                   │  │
│   │    多个 from_rank → 一个 to_rank (需要 merge)                       │  │
│   │    _merge_rank_sharded_kv_payloads()                                │  │
│   │                                                                      │  │
│   │  from_tp < to_tp:                                                   │  │
│   │    一个 from_rank → 多个 to_rank (需要 slice)                       │  │
│   │    _slice_rank_sharded_kv_payload()                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 五、请求处理流程

### 5.1 execute_model 完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      execute_model 完整流程                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   execute_model(req: OmniDiffusionRequest)                                  │
│   │                                                                         │
│   ├── 1. 前置检查                                                           │
│   │   ├── assert pipeline is not None                                      │
│   │   └── assert len(req.prompts) > 0                                      │
│   │                                                                         │
│   ├── 2. 选择梯度上下文 (HSDP 兼容性)                                       │
│   │   ├── use_hsdp → torch.no_grad()                                       │
│   │   └── otherwise → torch.inference_mode()                               │
│   │                                                                         │
│   ├── 3. 接收跨阶段 KV Cache (如果需要)                                     │
│   │   └── kv_transfer_manager.receive_multi_kv_cache_distributed(req)      │
│   │                                                                         │
│   ├── 4. 初始化随机数生成器                                                 │
│   │   └── if seed is not None:                                             │
│   │       generator = torch.Generator(device).manual_seed(seed)            │
│   │                                                                         │
│   ├── 5. 刷新 Cache Backend                                                 │
│   │   └── if cache_backend enabled:                                        │
│   │       cache_backend.refresh(pipeline, num_inference_steps)             │
│   │                                                                         │
│   ├── 6. 重置内存统计 (rank 0 only)                                         │
│   │   └── current_omni_platform.reset_peak_memory_stats()                  │
│   │                                                                         │
│   ├── 7. 设置 Forward Context                                               │
│   │   └── with set_forward_context(vllm_config, od_config):                │
│   │                                                                         │
│   ├── 8. 执行 Pipeline Forward                                              │
│   │   └── with record_function("pipeline_forward"):                        │
│   │           output = pipeline.forward(req)                                │
│   │                                                                         │
│   ├── 9. 记录峰值内存 (rank 0 only)                                         │
│   │   └── _record_peak_memory(output)                                      │
│   │                                                                         │
│   ├── 10. 打印 Cache 总结 (可选)                                            │
│   │   └── if enable_cache_dit_summary:                                     │
│   │           cache_summary(pipeline, details=True)                        │
│   │                                                                         │
│   └── 11. 返回输出                                                          │
│       └── return output                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 execute_stepwise 完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     execute_stepwise 完整流程                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   execute_stepwise(scheduler_output: DiffusionSchedulerOutput)              │
│   │                                                                         │
│   ├── 1. 前置检查                                                           │
│   │   ├── assert pipeline is not None                                      │
│   │   └── assert supports_step_mode()                                      │
│   │                                                                         │
│   ├── 2. 验证配置兼容性                                                     │
│   │   └── assert cache_backend is None  # Step mode 不支持 cache backend  │
│   │                                                                         │
│   ├── 3. 选择梯度上下文                                                     │
│   │   └── 同 execute_model                                                 │
│   │                                                                         │
│   ├── 4. 更新请求状态                                                       │
│   │   state, is_new_request = _update_states(scheduler_output)             │
│   │   │                                                                     │
│   │   ├── 清理已完成的请求                                                  │
│   │   │   └── for req_id in finished_req_ids:                              │
│   │   │           state_cache.pop(req_id, None)                            │
│   │   │                                                                     │
│   │   ├── 获取/创建当前请求状态                                             │
│   │   │   ├── if new request:                                              │
│   │   │   │   state = DiffusionRequestState(...)                          │
│   │   │   │   state_cache[req_id] = state                                  │
│   │   │   └── else:                                                        │
│   │   │       state = state_cache[req_id]                                  │
│   │   │                                                                     │
│   │   └── return state, is_new_request                                     │
│   │                                                                         │
│   ├── 5. 初始化新请求 (如果需要)                                            │
│   │   ├── 初始化随机数生成器                                                │
│   │   └── pipeline.prepare_encode(state)                                   │
│   │                                                                         │
│   ├── 6. 设置 Forward Context                                               │
│   │   └── with set_forward_context(...):                                   │
│   │                                                                         │
│   ├── 7. 执行去噪步骤                                                       │
│   │   noise_pred = pipeline.denoise_step(state)                            │
│   │                                                                         │
│   ├── 8. 处理中断情况 (CFG Parallel)                                        │
│   │   if noise_pred is None and pipeline.interrupt:                        │
│   │       finished = True                                                  │
│   │       result = DiffusionOutput(error="interrupted")                    │
│   │                                                                         │
│   ├── 9. 正常步骤                                                           │
│   │   ├── pipeline.step_scheduler(state, noise_pred)                       │
│   │   └── finished = state.denoise_completed                               │
│   │                                                                         │
│   ├── 10. 解码 (如果完成)                                                   │
│   │   if finished:                                                         │
│   │       result = pipeline.post_decode(state)                             │
│   │                                                                         │
│   ├── 11. 更新后置状态                                                      │
│   │   _update_states_after(state, finished)                               │
│   │   └── if finished: state_cache.pop(req_id)                             │
│   │                                                                         │
│   └── 12. 返回结果                                                          │
│       return RunnerOutput(                                                  │
│           req_id=state.req_id,                                              │
│           step_index=state.step_index,                                      │
│           finished=finished,                                                │
│           result=result                                                     │
│       )                                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 六、关键配置项

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          关键配置项                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   OmniDiffusionConfig 关键字段:                                             │
│   ─────────────────────────────────                                        │
│                                                                             │
│   # 模型加载                                                                │
│   model: str                      # 模型路径或名称                          │
│   model_class_name: str           # Pipeline 类名                           │
│   diffusion_load_format: str      # 加载格式                                │
│   enforce_eager: bool             # 是否禁用 torch.compile                  │
│                                                                             │
│   # CPU Offload                                                             │
│   enable_cpu_offload: bool        # 启用 CPU offload                        │
│   enable_layerwise_offload: bool  # 启用逐层 offload                        │
│                                                                             │
│   # Cache-DiT                                                               │
│   cache_backend: str              # Cache 后端名称 ("cache_dit")            │
│   cache_config: dict              # Cache 配置                              │
│   enable_cache_dit_summary: bool  # 打印 cache 总结                         │
│                                                                             │
│   # Step-wise 执行                                                          │
│   step_execution: bool            # 启用 step-wise 模式                     │
│                                                                             │
│   # 分布式                                                                   │
│   parallel_config: ParallelConfig # 并行配置                                │
│   ├── tensor_parallel_size: int                                            │
│   ├── sequence_parallel_size: int                                          │
│   ├── cfg_parallel_size: int                                               │
│   └── use_hsdp: bool              # 使用 HSDP                              │
│                                                                             │
│   # Sleep Mode                                                              │
│   enable_sleep_mode: bool         # 启用 sleep mode                         │
│                                                                             │
│   # LoRA                                                                    │
│   max_cpu_loras: int              # 最大 CPU LoRA 数量                      │
│   lora_path: str                  # LoRA 路径                               │
│   lora_scale: float               # LoRA 缩放                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 七、与 Worker 的协作

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DiffusionWorker 与 DiffusionModelRunner 协作            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DiffusionWorker (基础设施层)                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  职责:                                                               │  │
│   │  - 设备初始化 (GPU 设置)                                             │  │
│   │  - 分布式环境 (NCCL, TP, SP)                                         │  │
│   │  - 进程管理 (sleep/wake)                                             │  │
│   │  - LoRA 管理                                                         │  │
│   │  - Profiler 管理                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                   │                                        │
│                                   │ 持有                                    │
│                                   ▼                                        │
│   DiffusionModelRunner (模型层)                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  职责:                                                               │  │
│   │  - 模型加载与编译                                                    │  │
│   │  - Cache Backend 管理                                                │  │
│   │  - Offload Backend 管理                                              │  │
│   │  - 推理执行                                                          │  │
│   │  - 跨阶段数据传输                                                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   调用流程:                                                                  │
│   ─────────                                                                 │
│   Worker.execute_model(req, od_config)                                      │
│       │                                                                     │
│       ├── 1. 设置 LoRA adapter (如果有)                                    │
│       │   └── lora_manager.set_active_adapter(lora_request)               │
│       │                                                                     │
│       ├── 2. 获取 profiler context                                         │
│       │   └── ctx = profiler.annotate_context_manager("diffusion_forward") │
│       │                                                                     │
│       ├── 3. 调用 ModelRunner                                              │
│       │   └── with ctx:                                                    │
│       │           output = model_runner.execute_model(req)                 │
│       │                                                                     │
│       └── 4. 更新 profiler                                                 │
│           └── profiler.step()                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 八、总结

### 核心设计要点

| 组件 | 职责 | 关键技术 |
|------|------|----------|
| **DiffusionModelRunner** | 模型加载与推理 | Diffusers, torch.compile, Cache-DiT |
| **OmniConnectorModelRunnerMixin** | 跨阶段数据传输 | Connector, async chunk, KV transfer |
| **cache_backend** | 计算加速 | Cache-DiT 激活缓存 |
| **offload_backend** | 内存优化 | CPU offload, layerwise offload |
| **kv_transfer_manager** | KV Cache 跨阶段复用 | Connector + rank-aware routing |

### 两种执行模式选择

- **execute_model**: 简单场景，一次调用完成全部推理
- **execute_stepwise**: 复杂调度场景，支持抢占、多请求交错

### 性能优化手段

1. **torch.compile**: 编译优化 (非 eager mode)
2. **Cache-DiT**: 缓存中间激活，减少重复计算
3. **CPU Offload**: 将部分模块放 CPU，减少 GPU 显存
4. **KV Cache Transfer**: 跨阶段复用 KV Cache
