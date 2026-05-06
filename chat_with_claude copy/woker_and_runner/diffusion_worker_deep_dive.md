# DiffusionWorker 深度解析

## 一、核心初始化流程

### 1.1 初始化入口

```python
def __init__(
    self,
    local_rank: int,
    rank: int,
    od_config: OmniDiffusionConfig,
    skip_load_model: bool = False,
):
```

初始化流程包含**四个核心步骤**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DiffusionWorker.__init__()                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│ 1. init_device()  │───►│ 2. 创建           │───►│ 3. load_model()   │
│                   │    │    ModelRunner     │    │    [可选]         │
│ 设备/分布式初始化 │    │                   │    │                   │
└───────────────────┘    └───────────────────┘    └─────────┬─────────┘
                                                            │
                                                            ▼
                                                 ┌───────────────────┐
                                                 │ 4. init_lora_     │
                                                 │    manager()      │
                                                 │    [可选]         │
                                                 └───────────────────┘
```

---

## 二、init_device() - 设备与分布式初始化

### 2.1 职责

**最基础的初始化步骤**，负责：
1. 设置分布式环境变量
2. 初始化 CUDA 设备
3. 创建 VllmConfig
4. 初始化分布式通信（NCCL）
5. 初始化模型并行组

### 2.2 详细流程

```python
def init_device(self) -> None:
    world_size = self.od_config.num_gpus
    rank = self.rank

    # ========== Step 1: 设置分布式环境变量 ==========
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(self.od_config.master_port)
    os.environ["LOCAL_RANK"] = str(self.local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # ========== Step 2: 初始化 CUDA 设备 ==========
    self.device = current_omni_platform.get_torch_device(rank)
    current_omni_platform.set_device(self.device)

    # ========== Step 3: 创建 VllmConfig ==========
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(),
        device_config=DeviceConfig(device=self.device),
    )
    # 设置并行配置
    vllm_config.parallel_config.tensor_parallel_size = ...
    vllm_config.parallel_config.data_parallel_size = ...
    # ...

    # ========== Step 4 & 5: 初始化分布式环境 ==========
    with set_forward_context(...), set_current_vllm_config(self.vllm_config):
        # 初始化 NCCL 通信
        init_distributed_environment(world_size=world_size, rank=rank)

        # 初始化模型并行组
        initialize_model_parallel(
            data_parallel_size=...,
            cfg_parallel_size=...,
            sequence_parallel_size=...,
            ulysses_degree=...,
            ring_degree=...,
            tensor_parallel_size=...,
            pipeline_parallel_size=...,
            ...
        )
        init_workspace_manager(self.device)
```

### 2.3 分布式并行组初始化

`initialize_model_parallel()` 创建多个正交的进程组：

```
假设 16 个 GPU，配置为：
- data_parallel_size = 2
- cfg_parallel_size = 2  
- sequence_parallel_size = 2 (ulysses=2, ring=1)
- pipeline_parallel_size = 2

并行组划分：
┌─────────────────────────────────────────────────────────────────────────┐
│ DP Groups (8 组，每组 2 个 GPU):                                        │
│   [g0, g8], [g1, g9], [g2, g10], [g3, g11],                            │
│   [g4, g12], [g5, g13], [g6, g14], [g7, g15]                           │
├─────────────────────────────────────────────────────────────────────────┤
│ CFG Groups (8 组，每组 2 个 GPU):                                       │
│   [g0, g4], [g1, g5], [g2, g6], [g3, g7],                              │
│   [g8, g12], [g9, g13], [g10, g14], [g11, g15]                         │
├─────────────────────────────────────────────────────────────────────────┤
│ SP Groups (8 组，每组 2 个 GPU):                                        │
│   [g0, g1], [g2, g3], [g4, g5], [g6, g7],                              │
│   [g8, g9], [g10, g11], [g12, g13], [g14, g15]                         │
├─────────────────────────────────────────────────────────────────────────┤
│ PP Groups (8 组，每组 2 个 GPU):                                        │
│   [g0, g2], [g4, g6], [g8, g10], [g12, g14],                           │
│   [g1, g3], [g5, g7], [g9, g11], [g13, g15]                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 ForwardContext 的作用

```python
with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
    # 初始化分布式环境
    init_distributed_environment(...)
    initialize_model_parallel(...)
```

**ForwardContext** 是一个线程本地上下文，存储当前前向传播的配置信息：

```python
@dataclass
class ForwardContext:
    vllm_config: VllmConfig | None = None
    omni_diffusion_config: OmniDiffusionConfig | None = None
    attn_metadata: dict[str, AttentionMetadata] | None = None

    # Sequence Parallel 相关
    sp_padding_size: int = 0
    sp_original_seq_len: int | None = None
    sp_plan_hooks_applied: bool = False
    _sp_shard_depth: int = 0
```

**用途**：
- 在模型前向传播时访问配置
- 支持 Sequence Parallel 的动态控制
- 存储注意力元数据

---

## 三、创建 DiffusionModelRunner

### 3.1 Runner 的职责

```python
self.model_runner = DiffusionModelRunner(
    vllm_config=self.vllm_config,
    od_config=self.od_config,
    device=self.device,
)
```

**DiffusionModelRunner** 负责：
- 模型加载和编译
- 模型执行（前向传播）
- 缓存后端管理
- Offload 后端管理
- KV Transfer 管理

### 3.2 Runner 初始化

```python
class DiffusionModelRunner:
    def __init__(self, vllm_config, od_config, device):
        self.vllm_config = vllm_config
        self.od_config = od_config
        self.device = device
        self.pipeline = None              # 待加载的模型
        self.cache_backend = None         # 缓存后端
        self.offload_backend = None       # Offload 后端
        self.state_cache: dict = {}       # Stepwise 状态缓存

        # 初始化 KV Transfer 管理器
        self.kv_transfer_manager = OmniKVTransferManager.from_od_config(od_config)
```

---

## 四、load_model() - 模型加载

### 4.1 完整流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       DiffusionWorker.load_model()                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 设置加载设备                                                          │
│    load_device = "cpu" if enable_cpu_offload else str(self.device)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. 获取内存池上下文（用于 Sleep Mode）                                    │
│    memory_context = memory_pool_context_fn(tag="weights")               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. 创建 DiffusersPipelineLoader 并加载模型                               │
│    model_loader = DiffusersPipelineLoader(load_config, od_config)       │
│    pipeline = model_loader.load_model(...)                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. 应用 CPU Offloading（如果启用）                                       │
│    offload_backend = get_offload_backend(od_config)                     │
│    offload_backend.enable(pipeline)                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. 应用 torch.compile（如果非 eager 模式）                               │
│    transformer = regionally_compile(transformer, dynamic=True)          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. 设置缓存后端（Cache-DiT, TeaCache 等）                                │
│    cache_backend = get_cache_backend(cache_backend, cache_config)       │
│    cache_backend.enable(pipeline)                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 内存池上下文（Sleep Mode 支持）

```python
def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
    is_sleep_enabled = getattr(self.od_config, "enable_sleep_mode", False)
    if is_sleep_enabled:
        from vllm.device_allocator.cumem import CuMemAllocator
        allocator = CuMemAllocator.get_instance()
        return allocator.use_memory_pool(tag=tag)
    return nullcontext()
```

**作用**：当启用 Sleep Mode 时，模型权重被分配在 CuMem 内存池中，可以被卸载和重新加载。

### 4.3 模型加载格式

| load_format | 说明 |
|-------------|------|
| `"default"` | 自动检测，使用默认加载方式 |
| `"custom_pipeline"` | 从自定义 Pipeline 类初始化 |
| `"dummy"` | 跳过实际权重加载，用于测试 |
| `"diffusers"` | 使用 HuggingFace Diffusers 适配器 |

### 4.4 Offload 后端

```python
# 启用 CPU Offloading
if od_config.enable_cpu_offload:
    offload_backend.enable(pipeline)
    # DiT 和编码器互斥使用 GPU：
    # - 文本编码器在 GPU 时，DiT 在 CPU
    # - DiT 在 GPU 时，编码器在 CPU

# 启用 Layer-wise Offloading（更细粒度）
if od_config.enable_layerwise_offload:
    offload_backend.enable(pipeline)
    # 按 layer 级别进行 offload
```

### 4.5 torch.compile 优化

```python
if not od_config.enforce_eager:
    if platform.supports_torch_inductor():
        # 编码 transformer 以加速推理
        transformer = regionally_compile(transformer, dynamic=True)
```

### 4.6 缓存后端

```python
cache_backend = get_cache_backend(
    od_config.cache_backend,  # "tea_cache", "cache_dit", "none"
    od_config.cache_config
)

if cache_backend:
    cache_backend.enable(pipeline)
```

**支持的缓存后端**：
- `tea_cache`: TeaCache 加速
- `cache_dit`: Cache-DiT 特征缓存
- `deep_cache`: DeepCache

---

## 五、核心方法详解

### 5.1 execute_model() - 模型执行

```python
def execute_model(self, req: OmniDiffusionRequest, od_config: OmniDiffusionConfig) -> DiffusionOutput:
    """
    执行模型前向传播的核心方法。
    """
    assert self.model_runner is not None

    # ========== 1. LoRA 处理 ==========
    if self.lora_manager is not None:
        self.lora_manager.set_active_adapter(
            req.sampling_params.lora_request,
            req.sampling_params.lora_scale
        )

    # ========== 2. 性能分析上下文 ==========
    profiler = self._get_profiler()
    ctx = profiler.annotate_context_manager("diffusion_forward") if profiler else nullcontext()

    # ========== 3. 委托给 ModelRunner ==========
    with ctx:
        output = self.model_runner.execute_model(req)

    # ========== 4. Profiler 步进 ==========
    if profiler:
        profiler.step()

    return output
```

### 5.2 DiffusionModelRunner.execute_model() 详解

```python
def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
    # ========== 1. 选择梯度上下文 ==========
    # HSDP 需要 no_grad()，其他用 inference_mode() 更快
    use_hsdp = self.od_config.parallel_config.use_hsdp
    grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()

    with grad_context:
        # ========== 2. KV Cache 接收（多模态连接器） ==========
        self.kv_transfer_manager.receive_multi_kv_cache_distributed(req, ...)

        # ========== 3. 设置随机数生成器 ==========
        if req.sampling_params.generator is None and req.sampling_params.seed is not None:
            req.sampling_params.generator = torch.Generator(device=device).manual_seed(seed)

        # ========== 4. 刷新缓存上下文 ==========
        if self.cache_backend is not None:
            self.cache_backend.refresh(pipeline, num_inference_steps)

        # ========== 5. 重置内存统计 ==========
        if is_primary:
            platform.reset_peak_memory_stats()

        # ========== 6. 执行前向传播 ==========
        with set_forward_context(vllm_config, omni_diffusion_config):
            output = self.pipeline.forward(req)

        # ========== 7. 记录峰值内存 ==========
        if is_primary:
            self._record_peak_memory(output)

        return output
```

### 5.3 generate() - 生成入口

```python
def generate(self, request: OmniDiffusionRequest) -> DiffusionOutput:
    """Generate output for the given requests."""
    return self.execute_model(request, self.od_config)
```

### 5.4 execute_stepwise() - 步进执行

用于细粒度控制和调度：

```python
def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> RunnerOutput:
    """
    执行单步扩散，支持细粒度调度。

    与 execute_model 的区别：
    - execute_model: 一次性完成所有扩散步骤
    - execute_stepwise: 每次只执行一个步骤，由调度器控制
    """
    with grad_context:
        # 更新请求状态
        state, is_new_request = self._update_states(scheduler_output)

        if is_new_request:
            # 新请求：执行编码
            self.pipeline.prepare_encode(state)

        # 执行去噪步骤
        noise_pred = self.pipeline.denoise_step(state)

        # 执行调度器步骤
        self.pipeline.step_scheduler(state, noise_pred)

        # 检查是否完成
        finished = state.denoise_completed
        if finished:
            result = self.pipeline.post_decode(state)

        return RunnerOutput(req_id=state.req_id, finished=finished, result=result)
```

---

## 六、Sleep/Wake 机制

### 6.1 设计目的

**Sleep Mode** 允许卸载模型权重以释放 GPU 内存，用于：
- 多模型共存（如 LLM + Diffusion）
- 动态资源分配
- 内存紧张时的临时释放

### 6.2 sleep() 实现

```python
def sleep(self, level: int = 1) -> bool:
    """
    休眠模式：
    - Level 1: 卸载权重到 CPU
    - Level 2: 完全丢弃权重（需要重新加载）
    """
    from vllm.device_allocator.cumem import CuMemAllocator
    allocator = CuMemAllocator.get_instance()

    # Level 2: 保存 buffers 到 CPU
    if level == 2 and self.model_runner is not None:
        # 清理 CUDA Graph
        if hasattr(self.model_runner, "graph_runners"):
            self.model_runner.graph_runners.clear()

        # 保存模型 buffers
        model = self.model_runner.pipeline
        self._sleep_saved_buffers = {
            name: buffer.cpu().clone()
            for name, buffer in model.named_buffers()
        }

    # 执行 sleep
    offload_tags = ("weights",) if level == 1 else tuple()
    allocator.sleep(offload_tags=offload_tags)

    # 清理缓存
    platform.empty_cache()
    platform.synchronize()

    return True
```

### 6.3 wake_up() 实现

```python
def wake_up(self, tags: list[str] | None = None) -> bool:
    """
    从休眠模式唤醒，重新加载模型权重。
    """
    from vllm.device_allocator.cumem import CuMemAllocator
    allocator = CuMemAllocator.get_instance()

    # 激活内存池
    allocator.wake_up(tags)
    platform.synchronize()

    # 恢复 buffers
    if len(self._sleep_saved_buffers) and self.model_runner is not None:
        model = self.model_runner.pipeline
        for name, buffer in model.named_buffers():
            if name in self._sleep_saved_buffers:
                buffer.data.copy_(self._sleep_saved_buffers[name].data)
        self._sleep_saved_buffers = {}

    return True
```

### 6.4 handle_sleep_task() / handle_wake_task()

处理来自主进程的 Sleep/Wake 任务：

```python
def handle_sleep_task(self, task: OmniSleepTask) -> OmniACK:
    try:
        # 同步 GPU
        platform.synchronize()
        usage_before = platform.get_current_memory_usage(self.device)

        # 执行 sleep
        self.sleep(level=task.level)

        # 多 GPU 同步
        if torch.distributed.is_initialized():
            t_freed = torch.tensor([float(real_freed)], device=self.device)
            torch.distributed.all_reduce(t_freed)

        # 仅 rank 0 返回 ACK
        if self.rank != 0:
            return None

        return OmniACK(
            task_id=task.task_id,
            status="SUCCESS",
            freed_bytes=real_freed,
            metadata={...}
        )
    except Exception as e:
        return OmniACK(task_id=task.task_id, status="ERROR", error_msg=str(e))
```

---

## 七、Worker 生命周期

### 7.1 完整生命周期图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Worker 生命周期                                │
└─────────────────────────────────────────────────────────────────────────┘

时间 ──────────────────────────────────────────────────────────────────────►

┌─────────────┐
│ 创建阶段    │
│             │
│ __init__()  │
│ ├─ init_device()        # 设备、分布式初始化
│ ├─ ModelRunner 创建     # Runner 实例化
│ ├─ load_model()         # 模型加载
│ └─ init_lora_manager()  # LoRA 初始化
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 运行阶段    │
│             │
│ busy_loop() │◄──────────────────────────────────────┐
│             │                                       │
│ ├─ recv_message()                                   │
│ ├─ execute_model()  ──► 返回结果                    │
│ ├─ handle_sleep_task()                              │
│ ├─ handle_wake_task()                               │
│ └─ execute_rpc()                                    │
│             │                                       │
│             ├───────────────────────────────────────┘
│             │ (循环)
└──────┬──────┘
       │
       │ 收到 shutdown 消息
       ▼
┌─────────────┐
│ 关闭阶段    │
│             │
│ shutdown()  │
│ └─ destroy_distributed_env()  # 销毁分布式环境
└─────────────┘
```

### 7.2 状态转换

```
         ┌──────────────┐
         │   CREATED    │
         │  (刚创建)    │
         └──────┬───────┘
                │ init_device() 成功
                ▼
         ┌──────────────┐
         │   READY      │
         │  (模型已加载) │
         └──────┬───────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌──────────────┐ ┌──────────────┐
│   RUNNING    │ │   SLEEPING   │
│  (处理请求)  │ │  (权重卸载)  │
└──────┬───────┘ └──────┬───────┘
        │               │
        │  sleep()      │  wake_up()
        └───────────────┘
                │
                │ shutdown
                ▼
         ┌──────────────┐
         │   SHUTDOWN   │
         │   (已关闭)   │
         └──────────────┘
```

---

## 八、请求执行流程

### 8.1 完整请求流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          请求执行完整流程                                │
└─────────────────────────────────────────────────────────────────────────┘

主进程:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 创建 Request │────►│ MessageQueue │────►│ 等待结果     │
│              │     │  enqueue()   │     │  dequeue()   │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                            │ 共享内存
                            ▼
子进程 (Worker):
┌──────────────────────────────────────────────────────────────────────┐
│ worker_busy_loop()                                                   │
│                                                                      │
│   while _running:                                                    │
│       msg = mq.dequeue(timeout=1.0)  ◄─── 接收消息                   │
│       │                                                              │
│       ├─ if msg["type"] == "sleep":                                 │
│       │      handle_sleep_task() ──► return_result(ack)             │
│       │                                                              │
│       ├─ if msg["type"] == "wake_up":                               │
│       │      handle_wake_task() ──► return_result(ack)              │
│       │                                                              │
│       ├─ if msg["type"] == "rpc":                                   │
│       │      execute_rpc() ──► return_result(result)                │
│       │                                                              │
│       └─ else: (生成请求)                                            │
│              execute_model(msg) ──► return_result(output)           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.2 execute_model 内部流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DiffusionWorker.execute_model(req)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 设置 LoRA 适配器                                                     │
│    lora_manager.set_active_adapter(lora_request, lora_scale)           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. 创建 Profiler 上下文                                                 │
│    ctx = profiler.annotate_context_manager("diffusion_forward")        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. 委托给 ModelRunner                                                   │
│    with ctx:                                                            │
│        output = model_runner.execute_model(req)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. Profiler 步进                                                        │
│    profiler.step()                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                               返回 output
```

### 8.3 ModelRunner.execute_model 内部流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│               DiffusionModelRunner.execute_model(req)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│ 设置梯度上下文│          │ KV Cache 接收 │          │ 设置 Generator│
│ no_grad() 或  │          │ (多模态连接器)│          │ (随机种子)    │
│ inference_mode│          └───────────────┘          └───────────────┘
└───────┬───────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 刷新 Cache Backend (如果启用)                                           │
│ cache_backend.refresh(pipeline, num_inference_steps)                   │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 重置内存统计 (rank 0)                                                   │
│ platform.reset_peak_memory_stats()                                     │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 设置 ForwardContext                                                     │
│ with set_forward_context(vllm_config, omni_diffusion_config):          │
│     output = pipeline.forward(req)                                     │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 记录峰值内存 (rank 0)                                                   │
│ _record_peak_memory(output)                                            │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
                               返回 DiffusionOutput
```

---

## 九、关键组件交互图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DiffusionWorker                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      核心属性                                    │   │
│  │                                                                  │   │
│  │  device: torch.device        # CUDA 设备                        │   │
│  │  vllm_config: VllmConfig     # vLLM 配置                        │   │
│  │  od_config: OmniDiffusionConfig  # Diffusion 配置               │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐  │
│  │   model_runner    │  │   lora_manager    │  │    profiler       │  │
│  │                   │  │                   │  │                   │  │
│  │ DiffusionModel    │  │ DiffusionLoRA     │  │ WorkerProfiler    │  │
│  │ Runner            │  │ Manager           │  │                   │  │
│  │                   │  │                   │  │                   │  │
│  │ ├─ pipeline       │  │ ├─ adapters       │  │                   │  │
│  │ ├─ cache_backend  │  │ ├─ lora_path      │  │                   │  │
│  │ ├─ offload_backend│  │ └─ lora_scale     │  │                   │  │
│  │ └─ kv_transfer    │  │                   │  │                   │  │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘  │
│           │                      │                       │             │
│           │                      │                       │             │
│           ▼                      ▼                       ▼             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        GPU 设备                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 十、配置影响初始化

### 10.1 OmniDiffusionConfig 关键字段

| 字段 | 影响 |
|------|------|
| `model` | 模型路径，用于加载 |
| `model_class_name` | Pipeline 类名 |
| `dtype` | 模型精度 (bf16, fp16, fp32) |
| `parallel_config` | 并行配置 (TP, SP, PP, DP 等) |
| `enable_cpu_offload` | 启用 CPU Offloading |
| `enable_layerwise_offload` | 启用 Layer-wise Offloading |
| `cache_backend` | 缓存后端 (tea_cache, cache_dit) |
| `enforce_eager` | 禁用 torch.compile |
| `enable_sleep_mode` | 启用 Sleep Mode |
| `lora_path` | LoRA 适配器路径 |
| `quantization_config` | 量化配置 |

### 10.2 DiffusionParallelConfig 关键字段

| 字段 | 说明 |
|------|------|
| `tensor_parallel_size` | 张量并行度 |
| `sequence_parallel_size` | 序列并行度 |
| `ulysses_degree` | Ulysses 并行度 |
| `ring_degree` | Ring 并行度 |
| `pipeline_parallel_size` | 流水线并行度 |
| `data_parallel_size` | 数据并行度 |
| `cfg_parallel_size` | CFG 并行度 |
| `use_hsdp` | 启用 HSDP |

---

## 十一、总结

### 核心流程

```
初始化:
  init_device() → 创建 ModelRunner → load_model() → init_lora_manager()

请求执行:
  dequeue() → execute_model() → model_runner.execute_model() → pipeline.forward()

Sleep/Wake:
  sleep() → CuMemAllocator.sleep() → 卸载权重
  wake_up() → CuMemAllocator.wake_up() → 恢复权重

生命周期:
  CREATED → READY → (RUNNING ↔ SLEEPING) → SHUTDOWN
```

### 关键设计

1. **分层架构**：Worker 管理基础设施，Runner 管理模型
2. **ForwardContext**：线程本地上下文，存储前向传播配置
3. **分布式并行**：支持多种正交并行策略
4. **Sleep Mode**：支持动态内存管理
5. **缓存加速**：集成 Cache-DiT、TeaCache 等后端
