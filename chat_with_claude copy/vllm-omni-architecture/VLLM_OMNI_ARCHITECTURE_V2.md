# vLLM-Omni 架构详解（完整版）

## 目录

1. [概述](#概述)
2. [完整架构图](#完整架构图)
3. [核心组件详解](#核心组件详解)
4. [进程/线程/协程组织](#进程线程协程组织)
5. [通信机制](#通信机制)
6. [请求生命周期](#请求生命周期)
7. [组件生命周期](#组件生命周期)

---

## 概述

vLLM-Omni 是一个多阶段（multi-stage）的全方位（omnimodal）模型运行时，在 vLLM V1 架构基础上扩展，支持：
- **多阶段流水线**：Thinker AR 模型 -> Talker/TTS -> Diffusion
- **混合阶段类型**：LLM（自回归）和 Diffusion 模型可组合
- **分布式推理**：单阶段 CLI 模式支持跨进程/跨节点部署
- **KV Cache 传输**：阶段间 KV cache 高效传递

---

## 完整架构图

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    主进程 (Caller Process)                               │
│                                                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              主线程 (Main Thread)                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                        AsyncOmniEngine                                       │ │ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │ │ │
│  │  │  │  request_queue  │  │  output_queue   │  │    InputProcessor (vLLM)    │  │ │ │
│  │  │  │  (janus.Queue)  │  │  (janus.Queue)  │  │    - Tokenization           │  │ │ │
│  │  │  └────────┬────────┘  └────────┬────────┘  │    - Multimodal Processing  │  │ │ │
│  │  │           │                    │           └─────────────────────────────┘  │ │ │
│  │  │           │ sync_q.put()       │ sync_q.get()                               │ │ │
│  │  └───────────┼────────────────────┼───────────────────────────────────────────┼─┘ │
│  │              │                    │                                             │   │
│  └──────────────┼────────────────────┼─────────────────────────────────────────────┘   │
│                 │ spawn              │                                                   │
│                 ▼                    ▲                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                      Orchestrator 线程 (Background Thread)                        │  │
│  │                         独立 asyncio 事件循环                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                              Orchestrator                                   │  │  │
│  │  │  ┌──────────────────┐  ┌───────────────────┐  ┌──────────────────────────┐ │  │  │
│  │  │  │ request_async_q  │  │ output_async_q    │  │ request_states           │ │  │  │
│  │  │  │ (janus.AsyncQ)   │  │ (janus.AsyncQ)    │  │ dict[str, Orchestrator-  │ │  │  │
│  │  │  └────────┬─────────┘  └────────▲──────────┘  │      RequestState]       │ │  │  │
│  │  │           │                     │             └──────────────────────────┘ │  │  │
│  │  │           │                     │                                        │  │  │
│  │  │  ┌────────▼─────────────────────┴────────────────────────────────────────┐ │  │  │
│  │  │  │                asyncio Tasks (协程)                                   │ │  │  │
│  │  │  │  ┌──────────────────────┐  ┌────────────────────────────────────────┐ │ │  │  │
│  │  │  │  │  _request_handler()  │  │  _orchestration_output_handler()       │ │ │  │  │
│  │  │  │  │  - add_request       │  │  - 轮询所有阶段                        │ │ │  │  │
│  │  │  │  │  - abort             │  │  - 处理输出                            │ │ │  │  │
│  │  │  │  │  - collective_rpc    │  │  - 阶段间路由                          │ │ │  │  │
│  │  │  │  └──────────────────────┘  └────────────────────────────────────────┘ │ │  │  │
│  │  │  └────────────────────────────────────────────────────────────────────────┘ │  │  │
│  │  │                                                                              │  │  │
│  │  │  ┌────────────────────────────────────────────────────────────────────────┐ │  │  │
│  │  │  │                     stage_clients: list[Any]                            │ │  │  │
│  │  │  │  ┌────────────────────────┐  ┌────────────────────────────────────────┐│ │  │  │
│  │  │  │  │ StageEngineCoreClient  │  │     StageDiffusionClient               ││ │  │  │
│  │  │  │  │     (LLM 阶段)         │  │       (Diffusion 阶段)                  ││ │  │  │
│  │  │  │  │  继承 AsyncMPClient    │  │       ZMQ PUSH/PULL                    ││ │  │  │
│  │  │  │  └───────────┬────────────┘  └───────────────────┬────────────────────┘│ │  │  │
│  │  │  │              │ ZMQ PUSH/PULL                      │ ZMQ PUSH/PULL      │ │  │  │
│  │  │  └──────────────┼────────────────────────────────────┼────────────────────┘ │  │  │
│  │  └─────────────────┼────────────────────────────────────┼──────────────────────┘  │  │
│  │                    │                                    │                         │  │
│  │  ┌─────────────────┴────────────────────────────────────┴───────────────────────┐  │  │
│  │  │                output_processors: list[MultimodalOutputProcessor]            │  │  │
│  │  │  - 处理 EngineCoreOutput -> RequestOutput                                    │  │  │
│  │  │  - 累积多模态输出 (images, audio, latents)                                    │  │  │
│  │  └──────────────────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │                                       │
                                        │ ZMQ                                   │ ZMQ
                                        ▼                                       ▼
┌───────────────────────────────────────────────────┐   ┌────────────────────────────────────────────┐
│            StageEngineCoreProc 子进程              │   │         StageDiffusionProc 子进程          │
│                   (LLM 阶段)                       │   │            (Diffusion 阶段)                │
│                                                   │   │                                            │
│  ┌─────────────────────────────────────────────┐  │   │  ┌────────────────────────────────────────┐ │
│  │              EngineCore                      │  │   │  │          DiffusionEngine              │ │
│  │  ┌────────────────────────────────────────┐ │  │   │  │  ┌──────────────────────────────────┐ │ │
│  │  │         Scheduler (V1)                  │ │  │   │  │  │  DiffusionScheduler              │ │ │
│  │  │  ┌──────────────────────────────────┐  │ │  │   │  │  │  - RequestScheduler              │ │ │
│  │  │  │  OmniARScheduler /               │  │ │  │   │  │  │  - StepScheduler                 │ │ │
│  │  │  │  OmniGenerationScheduler         │  │ │  │   │  │  │  - waiting/running queues        │ │ │
│  │  │  │  - waiting/running queues        │  │ │  │   │  │  └──────────────────────────────────┘ │ │
│  │  │  │  - KV transfer tracking          │  │ │  │   │  │                                      │ │
│  │  │  │  - ChunkTransferAdapter          │  │ │  │   │  │  ┌──────────────────────────────────┐ │ │
│  │  │  └──────────────────────────────────┘  │ │  │   │  │  │  executor: DiffusionExecutor     │ │ │
│  │  │                                       │ │  │   │  │  │  - MultiprocDiffusionExecutor   │ │ │
│  │  │  ┌──────────────────────────────────┐  │ │  │   │  │  └──────────────────────────────────┘ │ │
│  │  │  │       Executor (GPUWorker)       │  │ │  │   │  └────────────────────────────────────────┘ │
│  │  │  │  - OmniGPUWorkerBase            │  │ │  │   │                                            │
│  │  │  │  - ModelRunner                   │  │ │  │   │  ┌────────────────────────────────────────┐ │
│  │  │  │  - OmniConnectorModelRunnerMixin│  │ │  │   │  │    ThreadPoolExecutor (1 worker)      │ │
│  │  │  └──────────────────────────────────┘  │ │  │   │  │    - 阻塞 DiffusionEngine.step()       │ │
│  │  └────────────────────────────────────────┘  │  │   │  └────────────────────────────────────────┘ │
│  │                                               │  │   │                                            │
│  │  ┌─────────────────────────────────────────┐  │  │   │  ┌────────────────────────────────────────┐ │
│  │  │      OmniKVTransferManager              │  │  │   │  │           ZMQ async loop               │ │
│  │  │  - KV cache 提取/传输                   │  │  │   │  │    - add_request 处理                 │ │
│  │  │  - Connector (Mooncake/ZMQ/SHM)         │  │  │   │  │    - abort 处理                       │ │
│  │  └─────────────────────────────────────────┘  │  │   │  │    - collective_rpc 处理              │ │
│  └───────────────────────────────────────────────┘  │   │  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘   └────────────────────────────────────────────┘
                                                                    │
                                                                    │ MessageQueue (SHM)
                                                                    ▼
                                                        ┌─────────────────────────────┐
                                                        │    DiffusionWorker 子进程    │
                                                        │      (每个 GPU 一个)         │
                                                        │                             │
                                                        │  ┌───────────────────────┐  │
                                                        │  │  DiffusionModelRunner │  │
                                                        │  │  - 模型加载           │  │
                                                        │  │  - 编译/缓存          │  │
                                                        │  │  - 执行               │  │
                                                        │  └───────────────────────┘  │
                                                        │                             │
                                                        │  ┌───────────────────────┐  │
                                                        │  │  DiffusionLoRAManager │  │
                                                        │  └───────────────────────┘  │
                                                        │                             │
                                                        │  ┌───────────────────────┐  │
                                                        │  │  WorkerProc           │  │
                                                        │  │  - busy loop          │  │
                                                        │  │  - RPC 处理           │  │
                                                        │  └───────────────────────┘  │
                                                        └─────────────────────────────┘
```

### 单进程 Inline 模式（仅单个 Diffusion 阶段时）

```
┌─────────────────────────────────────────────────────────────────┐
│                         主进程                                   │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Orchestrator 线程                              │ │
│  │                                                            │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │         InlineStageDiffusionClient                    │  │ │
│  │  │         (无 ZMQ 开销)                                  │  │ │
│  │  │                                                       │  │ │
│  │  │  ┌─────────────────────────────────────────────────┐ │  │ │
│  │  │  │  DiffusionEngine (进程内)                        │ │  │ │
│  │  │  │                                                  │ │  │ │
│  │  │  │  ┌────────────────────────────────────────────┐ │ │  │ │
│  │  │  │  │  ThreadPoolExecutor (1 worker)             │ │ │  │ │
│  │  │  │  │  - 阻塞 diffusion 操作                      │ │ │  │ │
│  │  │  │  └────────────────────────────────────────────┘ │ │  │ │
│  │  │  │                                                  │ │  │ │
│  │  │  │  ┌────────────────────────────────────────────┐ │ │  │ │
│  │  │  │  │  MultiprocDiffusionExecutor                │ │ │  │ │
│  │  │  │  │  - 仍然使用子进程 workers                   │ │ │  │ │
│  │  │  │  └────────────────────────────────────────────┘ │ │  │ │
│  │  │  └─────────────────────────────────────────────────┘ │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              │ MessageQueue (SHM)               │
│                              ▼                                  │
│                 ┌─────────────────────────────┐                │
│                 │  DiffusionWorker 子进程      │                │
│                 └─────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心组件详解

### 1. AsyncOmniEngine

**文件位置:** `vllm_omni/engine/async_omni_engine.py`

**角色:** 用户 API 入口，运行在调用者主线程中的轻量级代理。

**进程/线程:** 与调用者同一进程、同一线程。

**职责:**
- 解析 YAML 格式的阶段配置文件
- 启动 Orchestrator 后台线程
- 通过 janus 队列与 Orchestrator 通信
- 提供同步/异步 API

**关键属性:**
```python
class AsyncOmniEngine:
    # 通信队列
    request_queue: janus.Queue[dict]      # 同步端 -> Orchestrator
    output_queue: janus.Queue[dict]       # Orchestrator -> 同步端
    rpc_output_queue: janus.Queue[dict]   # RPC 响应队列

    # 线程管理
    orchestrator_thread: threading.Thread  # 后台线程

    # 阶段信息
    stage_clients: list[Any]              # 阶段客户端列表
    stage_vllm_configs: list[Any]         # 每阶段配置
    output_processors: list[Any]          # 输出处理器
    input_processor: InputProcessor       # Stage-0 输入处理器
```

**公共 API:**
```python
# 请求管理
def add_request(request_id, prompt, sampling_params_list, ...) -> None
async def add_request_async(...) -> None
def add_streaming_update(...) -> None

# 输出获取
def try_get_output(timeout) -> dict | None
async def try_get_output_async() -> dict | None

# 控制
def abort(request_ids) -> None
def collective_rpc(method, timeout, args, kwargs, stage_ids) -> list

# 生命周期
def shutdown() -> None
def is_alive() -> bool
```

**生命周期:**
1. `__init__`: 解析配置、启动 Orchestrator 线程、等待就绪
2. **运行时**: 处理用户请求，转发到 Orchestrator
3. `shutdown`: 发送关闭信号，等待线程退出

---

### 2. Orchestrator

**文件位置:** `vllm_omni/engine/orchestrator.py`

**角色:** 中央协调器，管理多阶段流水线。

**进程/线程:** 与 AsyncOmniEngine 同进程，运行在**独立后台线程**中，拥有独立的 **asyncio 事件循环**。

**职责:**
- 持有所有阶段客户端和输出处理器
- 管理请求状态（`OrchestratorRequestState`）
- 在阶段之间路由输出
- 处理 CFG 伴随请求
- 支持 PD disaggregation

**关键属性:**
```python
class Orchestrator:
    # 通信队列 (来自 janus.Queue 的异步端)
    request_async_queue: janus.AsyncQueue
    output_async_queue: janus.AsyncQueue
    rpc_async_queue: janus.AsyncQueue

    # 阶段管理
    num_stages: int
    stage_clients: list[Any]          # StageEngineCoreClient 或 StageDiffusionClient
    output_processors: list[Any]      # MultimodalOutputProcessor 列表
    stage_vllm_configs: list[Any]

    # 请求状态
    request_states: dict[str, OrchestratorRequestState]

    # CFG 支持
    _cfg_tracker: CfgCompanionTracker

    # PD disaggregation
    _pd_pair: tuple[int, int] | None  # (prefill_idx, decode_idx)
    _pd_kv_params: dict[str, Any]

    # 控制
    _shutdown_event: asyncio.Event
    _fatal_error: str | None
```

**主要协程任务:**
```python
async def run() -> None:
    # 启动两个并发任务
    request_task = asyncio.create_task(self._request_handler())
    output_task = asyncio.create_task(self._orchestration_output_handler())
    await asyncio.gather(request_task, output_task)

async def _request_handler() -> None:
    # 处理来自主线程的消息
    while True:
        msg = await self.request_async_queue.get()
        if msg["type"] == "add_request":
            await self._handle_add_request(msg)
        elif msg["type"] == "abort":
            await self._handle_abort(msg)
        elif msg["type"] == "collective_rpc":
            await self._handle_collective_rpc(msg)
        elif msg["type"] == "shutdown":
            break

async def _orchestration_loop() -> None:
    # 轮询所有阶段的输出
    while not self._shutdown_event.is_set():
        for stage_id in range(self.num_stages):
            # 1. 轮询原始输出
            raw_outputs = await self._poll_stage_raw(stage_id)

            # 2. 处理 KV ready 信号
            await self._handle_kv_ready_raw_outputs(stage_id, raw_outputs)

            # 3. 通过 OutputProcessor 处理
            request_outputs = await self._process_stage_outputs(stage_id, raw_outputs)

            # 4. 路由输出
            for output in request_outputs:
                await self._route_output(stage_id, output, req_state, metrics)
```

**生命周期:**
1. 在 `_bootstrap_orchestrator` 中创建，运行在独立线程
2. 初始化阶段客户端
3. 进入 `run()` 事件循环
4. 收到 shutdown 信号后清理并退出

---

### 3. StageEngineCoreClient (LLM 阶段客户端)

**文件位置:** `vllm_omni/engine/stage_engine_core_client.py`

**角色:** LLM/AR 阶段的客户端，继承自 vLLM 的 `AsyncMPClient`。

**进程架构:**
- 客户端运行在 **Orchestrator 线程**中
- Engine Core 运行在**独立子进程**中（`StageEngineCoreProc`）
- 通过 **ZMQ sockets** 通信

**类层次:**
```python
class StageEngineCoreClientBase:
    # 共享的阶段感知行为

class StageEngineCoreClient(StageEngineCoreClientBase, AsyncMPClient):
    # 标准单引擎客户端

class DPLBStageEngineCoreClient(StageEngineCoreClientBase, DPLBAsyncMPClient):
    # 数据并行负载均衡客户端
```

**关键属性:**
```python
class StageEngineCoreClientBase:
    # 阶段元数据
    stage_id: int
    stage_type: str                      # "llm"
    engine_output_type: str
    final_output: bool
    final_output_type: str               # "text", "audio", "image"
    default_sampling_params: Any
    custom_process_input_func: Callable
    engine_input_source: list[int]       # 输入来自哪个阶段

    # 引擎输出缓存
    engine_outputs: Any

    # 进程管理
    _proc: multiprocessing.Process

    # KV 传输
    _omni_kv_config: dict
    _kv_sender_host: str
    _kv_sender_info: dict
```

**关键方法:**
```python
async def add_request_async(request: EngineCoreRequest) -> None
async def get_output_async() -> EngineCoreOutputs
async def abort_requests_async(request_ids: list) -> None
async def collective_rpc_async(method, timeout, args, kwargs) -> Any

# 阶段间数据传输
def set_engine_outputs(outputs) -> None
def process_engine_inputs(stage_list, prompt, streaming_context) -> list
def get_kv_sender_info() -> dict | None
```

---

### 4. StageEngineCoreProc (LLM Engine Core 子进程)

**文件位置:** `vllm_omni/engine/stage_engine_core_proc.py`

**角色:** 运行 LLM EngineCore 忙循环的子进程，继承自 vLLM 的 `EngineCoreProc`。

**进程:** 运行在**独立子进程**中。

**生命周期:**
```python
def spawn_stage_core(vllm_config, executor_class, log_stats):
    # 1. 分配 ZMQ 地址
    addresses = get_engine_zmq_addresses(vllm_config)
    handshake_address = get_open_zmq_ipc_path()

    # 2. 启动子进程
    proc = ctx.Process(
        target=StageEngineCoreProc.run_stage_core,
        kwargs={
            "vllm_config": vllm_config,
            "executor_class": executor_class,
            "handshake_address": handshake_address,
            ...
        },
    )
    proc.start()
    return addresses, proc, handshake_address

def complete_stage_handshake(proc, handshake_address, addresses, vllm_config, timeout):
    # 3. HELLO 握手
    identity, msg = _recv(poller, handshake_socket, proc, "HELLO", timeout)

    # 4. 发送 INIT (地址信息)
    handshake_socket.send_multipart([identity, msgspec.msgpack.encode(init_payload)])

    # 5. READY 握手 (num_gpu_blocks)
    identity, msg = _recv(poller, handshake_socket, proc, "READY", timeout)
    vllm_config.cache_config.num_gpu_blocks = msg.get("num_gpu_blocks")

def run_stage_core(*args, **kwargs):
    # 6. 运行忙循环
    engine_core = StageEngineCoreProc(...)
    engine_core.run_busy_loop()
```

---

### 5. StageDiffusionClient (Diffusion 阶段客户端 - 进程外)

**文件位置:** `vllm_omni/diffusion/stage_diffusion_client.py`

**角色:** Diffusion 阶段的客户端，通过 ZMQ 与 `StageDiffusionProc` 通信。

**进程架构:**
- 客户端运行在 **Orchestrator 线程**中
- Diffusion 引擎运行在**独立子进程**中

**通信模式:**
```
[Orchestrator 线程]                    [子进程]
StageDiffusionClient ──ZMQ PUSH──> StageDiffusionProc
                     <──ZMQ PULL──   (DiffusionEngine)
```

**关键属性:**
```python
class StageDiffusionClient:
    stage_type: str = "diffusion"
    stage_id: int
    final_output: bool
    final_output_type: str

    # 进程管理
    _proc: multiprocessing.Process
    _owns_process: bool

    # ZMQ 通信
    _zmq_ctx: zmq.Context
    _request_socket: zmq.Socket    # PUSH
    _response_socket: zmq.Socket   # PULL

    # 序列化
    _encoder: OmniMsgpackEncoder
    _decoder: OmniMsgpackDecoder

    # 输出队列
    _output_queue: asyncio.Queue[OmniRequestOutput]

    # 进程监控
    _engine_dead: bool
```

**关键方法:**
```python
async def add_request_async(request_id, prompt, sampling_params, kv_sender_info) -> None
async def add_batch_request_async(request_id, prompts, sampling_params, kv_sender_info) -> None
def get_diffusion_output_nowait() -> OmniRequestOutput | None
async def abort_requests_async(request_ids) -> None
async def collective_rpc_async(method, timeout, args, kwargs) -> Any
```

---

### 6. InlineStageDiffusionClient (Diffusion 阶段客户端 - 进程内)

**文件位置:** `vllm_omni/diffusion/inline_stage_diffusion_client.py`

**角色:** 替代客户端，使用线程池在进程内运行 DiffusionEngine。

**使用场景:** 仅当有**单个 Diffusion 阶段**时使用（避免 ZMQ 开销）。

**进程架构:**
- 完全运行在 **Orchestrator 进程**中
- 使用 `ThreadPoolExecutor` 处理阻塞操作
- 通过 `asyncio.Queue` 通信（无 ZMQ）

**关键属性:**
```python
class InlineStageDiffusionClient:
    stage_type: str = "diffusion"

    # 引擎
    _engine: DiffusionEngine

    # 线程池
    _executor: ThreadPoolExecutor  # max_workers=1

    # 输出队列
    _output_queue: asyncio.Queue[OmniRequestOutput]

    # 任务追踪
    _tasks: dict[str, asyncio.Task]
```

---

### 7. StageDiffusionProc (Diffusion 子进程)

**文件位置:** `vllm_omni/diffusion/stage_diffusion_proc.py`

**角色:** Diffusion 推理的子进程入口点。

**进程:** 运行在**独立子进程**中。

**生命周期:**
```python
@classmethod
def run_diffusion_proc(cls, model, od_config, handshake_address, request_address, response_address):
    # 1. 创建实例
    proc = cls(model, od_config)

    # 2. 初始化引擎
    proc.initialize()  # 加载 DiffusionEngine

    # 3. 发送 READY 握手
    handshake_socket.send(msgspec.msgpack.encode({"status": "READY"}))

    # 4. 运行异步事件循环
    asyncio.run(proc.run_loop(request_address, response_address))
```

**异步事件循环:**
```python
async def run_loop(self, request_address, response_address):
    # 创建 ZMQ sockets
    request_socket = ctx.socket(zmq.PULL)  # 接收请求
    response_socket = ctx.socket(zmq.PUSH)  # 发送响应

    while True:
        raw = await request_socket.recv()
        msg = decoder.decode(raw)

        if msg["type"] == "add_request":
            # 派发到线程池
            task = asyncio.create_task(_dispatch_request(...))

        elif msg["type"] == "abort":
            self._engine.abort(msg["request_ids"])

        elif msg["type"] == "collective_rpc":
            result = await self._handle_collective_rpc(...)
            await response_socket.send(encoder.encode({"type": "rpc_result", ...}))

        elif msg["type"] == "shutdown":
            break
```

---

### 8. DiffusionEngine

**文件位置:** `vllm_omni/diffusion/diffusion_engine.py`

**角色:** 主要的 Diffusion 推理引擎。

**进程位置:**
- 运行在 `StageDiffusionProc` 子进程内，或
- 运行在 `InlineStageDiffusionClient` 内（进程内）

**关键属性:**
```python
class DiffusionEngine:
    # 配置
    od_config: OmniDiffusionConfig

    # 后处理函数
    post_process_func: Callable
    pre_process_func: Callable

    # 执行器
    executor: DiffusionExecutor  # MultiprocDiffusionExecutor

    # 调度器
    scheduler: SchedulerInterface  # RequestScheduler 或 StepScheduler

    # 执行模式
    step_execution: bool  # True: StepScheduler, False: RequestScheduler
```

**执行流程:**
```python
def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
    # 1. 预处理
    if self.pre_process_func:
        request = self.pre_process_func(request)

    # 2. 调度并执行
    output = self.add_req_and_wait_for_response(request)

    # 3. 后处理
    outputs = self.post_process_func(output.output) if self.post_process_func else output.output

    return [OmniRequestOutput.from_diffusion(...)]
```

---

### 9. Scheduler 组件

#### 9.1 Diffusion Schedulers

**文件位置:** `vllm_omni/diffusion/sched/`

**类层次:**
```
SchedulerInterface (ABC)
    └── _BaseScheduler
            ├── RequestScheduler    # 请求模式：一次完成
            └── StepScheduler       # 步进模式：逐步去噪
```

**SchedulerInterface 接口:**
```python
class SchedulerInterface(ABC):
    def initialize(self, od_config: OmniDiffusionConfig) -> None
    def add_request(self, request: OmniDiffusionRequest) -> str  # 返回 sched_req_id
    def schedule(self) -> DiffusionSchedulerOutput
    def update_from_output(self, sched_output, output) -> set[str]  # 返回 finished_req_ids
    def get_request_state(self, sched_req_id) -> DiffusionRequestState | None
    def has_requests(self) -> bool
    def finish_requests(self, sched_req_ids, status) -> None
    def close(self) -> None
```

**_BaseScheduler 状态:**
```python
class _BaseScheduler(SchedulerInterface):
    _request_states: dict[str, DiffusionRequestState]
    _request_id_to_sched_req_id: dict[str, str]
    _step_id: int
    _waiting: deque[str]       # WAITING 队列
    _running: list[str]        # RUNNING 队列
    _finished_req_ids: set[str]
    max_num_running_reqs: int  # 当前固定为 1
```

**RequestScheduler:** 单步完成整个 diffusion 请求

**StepScheduler:** 支持逐步去噪，追踪 `current_step` 和 `total_steps`

**进程位置:** 运行在 **DiffusionEngine** 内（Diffusion 子进程）

---

#### 9.2 LLM/AR Schedulers (OmniARScheduler)

**文件位置:** `vllm_omni/core/sched/omni_ar_scheduler.py`

**角色:** 自回归模型的调度器，扩展 vLLM V1 Scheduler。

**类层次:**
```python
class OmniARScheduler(OmniSchedulerMixin, VLLMScheduler):
    # 同步 AR 调度器

class OmniARAsyncScheduler(OmniARScheduler, AsyncVLLMScheduler):
    # 异步 AR 调度器

class OmniGenerationScheduler(OmniSchedulerMixin, VLLMScheduler):
    # Diffusion 快速路径调度器
```

**关键状态:**
```python
class OmniARScheduler:
    # KV 传输追踪
    requests_needing_kv_transfer: dict[str, dict]  # {seq_len, block_ids}
    waiting_for_transfer_free: set[str]
    active_kv_transfers: set[str]
    pending_stop_after_extraction: set[str]
    transfer_triggered_requests: set[str]

    # KV 传输条件
    kv_transfer_criteria: dict  # {"type": "prefill_finished" | "special_token", ...}

    # Chunk 传输适配器 (async_chunk 模式)
    chunk_transfer_adapter: OmniChunkTransferAdapter | None
```

**关键方法:**
```python
def schedule(self) -> SchedulerOutput:
    # 1. 处理待处理的 chunks
    if self.chunk_transfer_adapter:
        self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)

    # 2. 调用基类调度
    scheduler_output = super().schedule()

    # 3. 丰富输出为 OmniSchedulerOutput
    # ...

def update_from_output(self, scheduler_output, model_runner_output) -> tuple:
    # 1. 调用基类更新
    # 2. 处理 KV 传输触发
    for request in finished_requests:
        if self._process_kv_transfer_trigger(request, new_token_ids):
            # 标记停止
    # 3. 返回 (outputs, scheduler_stats)
```

**进程位置:** 运行在 **StageEngineCoreProc** 子进程内

---

### 10. DiffusionExecutor

**文件位置:** `vllm_omni/diffusion/executor/`

**类型:**
- `MultiprocDiffusionExecutor` - 多进程执行器（默认）
- `InlineDiffusionExecutor` - 进程内执行器

**MultiprocDiffusionExecutor:**

```python
class MultiprocDiffusionExecutor(DiffusionExecutor):
    uses_multiproc: bool = True

    # 通信
    _broadcast_mq: MessageQueue    # 广播到所有 workers
    _result_mq: MessageQueue       # 仅 rank-0 发送结果

    # 进程
    _processes: list[mp.Process]

    def _launch_workers(self, broadcast_handle, wake_events):
        for i in range(num_gpus):
            process = mp.Process(
                target=WorkerProc.worker_main,
                args=(i, od_config, writer, broadcast_handle, wake_event, ...),
            )
            process.start()
```

---

### 11. DiffusionWorker & WorkerProc

**文件位置:** `vllm_omni/diffusion/worker/diffusion_worker.py`

**角色:** GPU 工作进程，执行模型推理。

**进程:** 运行在**独立子进程**中（每个 GPU 一个）。

**DiffusionWorker:**
```python
class DiffusionWorker:
    local_rank: int
    rank: int
    device: torch.device
    vllm_config: VllmConfig

    # 模型运行器
    model_runner: DiffusionModelRunner

    # LoRA 管理
    lora_manager: DiffusionLoRAManager | None

    # 性能分析
    profiler: WorkerProfiler | None
```

**WorkerProc:**
```python
class WorkerProc:
    """在独立进程中运行 Worker 的包装器"""

    od_config: OmniDiffusionConfig
    gpu_id: int

    # IPC
    context: zmq.Context
    mq: MessageQueue              # 从主进程接收
    result_mq: MessageQueue       # 发送结果（仅 rank-0）

    # Worker 实例
    worker: DiffusionWorker

    def worker_busy_loop(self):
        while self._running:
            msg = self.mq.dequeue(timeout=1.0)

            if msg["type"] == "rpc":
                result, should_reply = self.execute_rpc(msg)
                if should_reply:
                    self.return_result(result)

            elif msg["type"] == "sleep":
                ack = self.worker.handle_sleep_task(task)
                self.return_result(ack)
```

---

### 12. DiffusionModelRunner

**文件位置:** `vllm_omni/diffusion/worker/diffusion_model_runner.py`

**角色:** 处理模型加载、编译、缓存和执行。

**关键功能:**
- 加载 diffusers 格式模型
- torch.compile 支持
- 缓存加速（Cache-DiT, TeaCache）
- CPU offloading
- KV transfer 管理

**关键属性:**
```python
class DiffusionModelRunner:
    vllm_config: VllmConfig
    od_config: OmniDiffusionConfig
    device: torch.device

    # 模型
    pipeline: Any  # diffusers pipeline

    # 缓存
    cache_backend: CacheBackend | None

    # Offloading
    offload_backend: OffloadBackend | None

    # KV 传输
    kv_transfer_manager: OmniKVTransferManager
```

---

### 13. OutputProcessor

**文件位置:** `vllm_omni/engine/output_processor.py`

**角色:** 将原始引擎输出处理为用户可见的 `RequestOutput`。

**进程位置:** 运行在 **Orchestrator 线程**中。

**关键类:**
```python
class OmniRequestState(RequestState):
    # 多模态输出累积
    mm_accumulated: dict[str, Any]

    def add_multimodal_tensor(self, payload, mm_type) -> None
    def _consolidate_multimodal_tensors(self) -> None
    def make_request_output(...) -> OmniRequestOutput | None

class MultimodalOutputProcessor(VLLMOutputProcessor):
    engine_core_output_type: str | None  # "image", "audio", "latent"

    def add_request(self, request, prompt, ...) -> None
    def process_outputs(self, engine_core_outputs, ...) -> OutputProcessorOutput
```

**输出模式:**
- `DELTA`: 每步增量输出
- `CUMULATIVE`: 累积输出
- `FINAL_ONLY`: 仅最终输出

---

### 14. OmniKVTransferManager

**文件位置:** `vllm_omni/distributed/omni_connectors/kv_transfer_manager.py`

**角色:** KV cache 跨阶段传输管理。

**关键功能:**
- KV cache 提取和序列化
- 通过 Connector 传输
- 接收和反序列化

**关键方法:**
```python
class OmniKVTransferManager:
    def handle_finished_requests_kv_transfer(self, requests, kv_cache_manager) -> list[str]
    def receive_kv_cache_for_request(self, request_id, sender_info, timeout) -> dict
    def apply_kv_cache_to_request(self, request, kv_data) -> None
```

**KVCacheTransferData:**
```python
@dataclass
class KVCacheTransferData:
    request_id: str
    layer_blocks: dict[str, Any]  # key_cache, value_cache
    block_ids: list[int]
    metadata: dict[str, Any]

    def to_bytes(self) -> bytes
    def to_gpu_tensor(self) -> torch.Tensor
    @staticmethod
    def from_bytes(raw) -> dict
```

---

### 15. OmniConnectorModelRunnerMixin

**文件位置:** `vllm_omni/worker/omni_connector_model_runner_mixin.py`

**角色:** 统一的数据平面通信 mixin，用于 ModelRunner。

**传输模式:**
- **full_payload_mode**: 完整负载传输
- **async_chunk**: 流式 chunk 传输
- **KV cache**: KV cache 传输

**关键状态:**
```python
class OmniConnectorModelRunnerMixin:
    # Connector
    _omni_connector: OmniConnectorBase | None
    _kv_transfer_manager: OmniKVTransferManager | None

    # 异步 I/O 状态
    _pending_load_reqs: dict[str, Any]
    _pending_save_reqs: dict[str, deque]

    # 本地缓存
    _local_stage_payload_cache: dict[str, dict]

    # 后台线程
    _recv_thread: threading.Thread
    _save_thread: threading.Thread
    _stop_event: threading.Event
```

---

### 16. OmniMasterServer

**文件位置:** `vllm_omni/engine/stage_engine_startup.py`

**角色:** 单阶段 CLI 模式下的协调服务器。

**职责:**
- 预分配 ZMQ 地址
- 阶段注册和配置分发
- 跨进程阶段发现

**关键方法:**
```python
class OmniMasterServer:
    def __init__(self, master_address, master_port, stage_ids):
        # 预分配每个阶段的地址
        for sid in stage_ids:
            self._allocations[sid] = StageAllocation(
                handshake_bind_address=...,
                input_bind_address=...,
                output_bind_address=...,
            )

    def register_stage_config(self, stage_id, stage_config, coordinator_addresses) -> None
    def get_stage_config(self, stage_id, timeout_s) -> Any
    def get_zmq_addresses(self, stage_id) -> StageAllocation
```

---

## 进程/线程/协程组织

### 详细对照表

| 组件 | 进程 | 线程 | 协程 | 通信方式 |
|------|------|------|------|----------|
| **AsyncOmniEngine** | 主进程 | 主线程 | - | - |
| **Orchestrator** | 主进程 | 后台线程 | 独立 asyncio 循环 | janus Queue |
| **StageEngineCoreClient** | 主进程 | Orchestrator 线程 | - | ZMQ |
| **StageEngineCoreProc** | 子进程 | 主线程 | - | ZMQ |
| **EngineCore (LLM)** | 子进程 | 主线程 | - | 内部调用 |
| **OmniARScheduler** | 子进程 | 主线程 | - | 内部调用 |
| **StageDiffusionClient** | 主进程 | Orchestrator 线程 | - | ZMQ |
| **InlineStageDiffusionClient** | 主进程 | Orchestrator 线程 | 线程池 | asyncio.Queue |
| **StageDiffusionProc** | 子进程 | 主线程 | 独立 asyncio 循环 | ZMQ |
| **DiffusionEngine** | 子进程 | 主线程/线程池 | - | 内部调用 |
| **MultiprocDiffusionExecutor** | 子进程 | 主线程 | - | MessageQueue |
| **DiffusionWorker** | 子进程 | 主线程 | - | MessageQueue |
| **WorkerProc** | 子进程 | 主线程 | - | MessageQueue |
| **OmniKVTransferManager** | 子进程 | 主线程 | - | Connector |
| **OmniConnectorModelRunnerMixin** | 子进程 | 主线程 + 后台 I/O 线程 | - | Connector |
| **MultimodalOutputProcessor** | 主进程 | Orchestrator 线程 | - | 内部调用 |
| **OmniMasterServer** | 主进程 | 主线程 | - | ZMQ (注册) |

### 线程类型

```
主进程
├── 主线程
│   └── AsyncOmniEngine (用户 API)
│
├── Orchestrator 线程 (后台)
│   ├── asyncio 事件循环
│   │   ├── _request_handler() 协程
│   │   └── _orchestration_output_handler() 协程
│   └── StageClients
│
└── OmniMasterServer 线程 (单阶段 CLI 模式)

StageEngineCoreProc 子进程
├── 主线程 (EngineCore 忙循环)
└── 后台监控线程

StageDiffusionProc 子进程
├── 主线程
│   └── asyncio 事件循环 (ZMQ 消息处理)
└── ThreadPoolExecutor (DiffusionEngine.step)

DiffusionWorker 子进程
├── 主线程 (WorkerProc busy loop)
└── OmniConnectorModelRunnerMixin 后台线程
    ├── _recv_loop (接收)
    └── _save_loop (发送)
```

---

## 通信机制

### 1. Janus Queues (线程间)

**用于:** AsyncOmniEngine ↔ Orchestrator

**类型:** `janus.Queue` - 同步/异步桥接

```
┌───────────────────┐                    ┌───────────────────┐
│    主线程          │                    │   Orchestrator    │
│                   │                    │     线程          │
│  sync_q.put() ────┼──> request_queue ──┼──> async_q.get()  │
│                   │                    │                   │
│  sync_q.get() <───┼─── output_queue ───┼─── async_q.put()  │
│                   │                    │                   │
└───────────────────┘                    └───────────────────┘
```

**消息类型:**
```python
# 请求
{"type": "add_request", "request_id": str, "prompt": Any, "sampling_params_list": list, ...}
{"type": "streaming_update", ...}
{"type": "abort", "request_ids": list}
{"type": "collective_rpc", "rpc_id": str, "method": str, "args": tuple, "kwargs": dict}

# 输出
{"type": "output", "request_id": str, "stage_id": int, "engine_outputs": Any, "finished": bool}
{"type": "error", "error": str, "request_id": str}
{"type": "collective_rpc_result", "rpc_id": str, "results": list}
```

---

### 2. ZMQ Sockets (进程间 - LLM 阶段)

**用于:** StageEngineCoreClient ↔ StageEngineCoreProc

**Socket 配置:**
```python
# 客户端 (Orchestrator 进程)
input_socket = ctx.socket(zmq.PUSH)   # 发送请求
output_socket = ctx.socket(zmq.PULL)  # 接收响应

# 服务端 (EngineCoreProc)
input_socket = ctx.socket(zmq.PULL)   # 接收请求
output_socket = ctx.socket(zmq.PUSH)  # 发送响应
```

**握手流程:**
```
StageEngineCoreProc                    StageEngineCoreClient
        │                                      │
        │──── HELLO (DEALER) ─────────────────>│
        │                                      │
        │<─── INIT (addresses, config) ────────│
        │                                      │
        │──── READY (num_gpu_blocks) ─────────>│
        │                                      │
        │         正常通信 (PUSH/PULL)          │
```

---

### 3. ZMQ Sockets (进程间 - Diffusion 阶段)

**用于:** StageDiffusionClient ↔ StageDiffusionProc

**消息格式:** Msgpack 编码

```python
# 请求
{
    "type": "add_request",
    "request_id": str,
    "prompt": Any,
    "sampling_params": dict,
    "kv_sender_info": dict | None,
}

# 批量请求
{
    "type": "add_batch_request",
    "request_id": str,
    "prompts": list,
    "sampling_params": dict,
}

# RPC
{
    "type": "collective_rpc",
    "rpc_id": str,
    "method": str,
    "args": list,
    "kwargs": dict,
}

# 响应
{"type": "result", "output": OmniRequestOutput}
{"type": "rpc_result", "rpc_id": str, "result": Any}
{"type": "error", "request_id": str, "error": str}
```

**死亡哨兵:**
```python
DIFFUSION_PROC_DEAD = b"DIFFUSION_PROC_DEAD"  # 进程崩溃时发送
```

---

### 4. MessageQueue (共享内存广播)

**用于:** MultiprocDiffusionExecutor ↔ DiffusionWorker 进程

**库:** vLLM 的 `shm_broadcast.MessageQueue`

**架构:**
```
                    ┌───────────────────────────────┐
                    │  MultiprocDiffusionExecutor   │
                    │                               │
                    │  _broadcast_mq (广播)          │
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │  Worker-0     │       │  Worker-1     │       │  Worker-N     │
    │  mq.dequeue() │       │  mq.dequeue() │       │  mq.dequeue() │
    └───────────────┘       └───────────────┘       └───────────────┘
            │
            ▼ (仅 rank-0)
    ┌───────────────┐
    │  result_mq    │ ────────> MultiprocDiffusionExecutor
    └───────────────┘
```

**代码示例:**
```python
# Executor 端
broadcast_mq = MessageQueue(
    n_reader=num_workers,
    n_local_reader=num_workers,
    local_reader_ranks=list(range(num_workers)),
)
broadcast_mq.enqueue(rpc_request)
response = result_mq.dequeue()

# Worker 端
mq = MessageQueue.create_from_handle(broadcast_handle, gpu_id)
msg = mq.dequeue(timeout=1.0)
if result_mq is not None:  # 仅 rank-0
    result_mq.enqueue(response)
```

---

### 5. POSIX 共享内存 (大型张量)

**用于:** Diffusion 输出传输

**实现:** `vllm_omni/diffusion/ipc.py`

**流程:**
```python
# 发送端 (Worker)
def pack_diffusion_output_shm(output: DiffusionOutput):
    if output.output is not None and output.output.numel() * element_size > 1MB:
        # 创建命名共享内存
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        # 复制张量
        np_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
        np_array[:] = tensor_bytes
        # 返回元数据
        output.shm_name = shm.name
        output.shm_shape = shape
        output.shm_dtype = dtype

# 接收端 (Executor)
def unpack_diffusion_output_shm(output: DiffusionOutput):
    if output.shm_name is not None:
        shm = shared_memory.SharedMemory(name=output.shm_name)
        tensor = torch.frombuffer(shm.buf, dtype=dtype).reshape(shape)
        # 复制到本地
        output.output = tensor.clone()
        shm.close()
        shm.unlink()  # 清理
```

---

### 6. KV Transfer (阶段间)

**用于:** 阶段间 KV cache 传输

**Connector 类型:**
- `MooncakeStoreConnector` - Mooncake 存储
- `MooncakeTransferEngineConnector` - RDMA 传输引擎
- `SharedMemoryConnector` - 本地共享内存
- `ZmqConnector` - ZMQ 传输

**传输流程:**
```
Stage 0 (Thinker)                         Stage 1 (Diffusion)
        │                                        │
        │  1. 提取 KV cache                       │
        │     OmniKVTransferManager              │
        │         .handle_finished_requests      │
        │                                        │
        │  2. 序列化                              │
        │     KVCacheTransferData.to_bytes()     │
        │         或 .to_gpu_tensor()            │
        │                                        │
        │  3. 发送 ──────────────────────────────>│
        │     connector.put(from_stage=0,        │
        │                   to_stage=1,          │
        │                   put_key=req_id,      │
        │                   data=kv_data)        │
        │                                        │
        │                              4. 接收    │
        │     connector.get(from_stage=0,        │
        │                   to_stage=1,          │
        │                   get_key=req_id)      │
        │                                        │
        │                              5. 反序列化 │
        │     KVCacheTransferData.from_bytes()   │
        │                                        │
        │                              6. 应用    │
        │     apply_kv_cache_to_request()        │
```

---

## 请求生命周期

### 单阶段 Diffusion 请求

```
1. 用户调用 AsyncOmniEngine.add_request()
   │
   ├─> InputProcessor.process_inputs()  [主线程]
   │   - Tokenization
   │   - Multimodal processing
   │   - Build EngineCoreRequest
   │
   ├─> OutputProcessor.add_request()  [主线程]
   │   - 注册请求状态
   │
   └─> request_queue.sync_q.put()  [主线程]
       - 发送到 Orchestrator

2. Orchestrator 处理  [Orchestrator 线程]
   │
   ├─> _request_handler() 协程接收消息
   │
   ├─> 创建 OrchestratorRequestState
   │
   └─> StageDiffusionClient.add_request_async()
       - ZMQ PUSH 发送到子进程

3. StageDiffusionProc 处理  [子进程]
   │
   ├─> ZMQ PULL 接收消息
   │
   ├─> asyncio.create_task(_dispatch_request())
   │   │
   │   └─> ThreadPoolExecutor.submit(DiffusionEngine.step)
   │       │
   │       ├─> RequestScheduler.add_request()
   │       ├─> RequestScheduler.schedule()
   │       ├─> MultiprocDiffusionExecutor.execute_request()
   │       │   │
   │       │   └─> MessageQueue.enqueue(request)
   │       │
   │       └─> DiffusionWorker.execute_model()
   │           │
   │           └─> DiffusionModelRunner.execute_model()
   │
   └─> ZMQ PUSH 发送结果

4. 结果返回  [Orchestrator 线程]
   │
   ├─> StageDiffusionClient.get_diffusion_output_nowait()
   │   - ZMQ PULL 接收结果
   │
   ├─> _orchestration_loop() 处理
   │   - _route_output()
   │   - final_output: 发送到 output_queue
   │
   └─> output_async_queue.put()

5. 用户获取结果  [主线程]
   │
   └─> AsyncOmniEngine.try_get_output()
       - output_queue.sync_q.get()
```

### 多阶段流水线 (Thinker -> Diffusion)

```
1. Stage 0 (Thinker AR 模型)
   │
   ├─> 用户 add_request()
   │
   ├─> Orchestrator 路由到 StageEngineCoreClient
   │
   ├─> ZMQ -> StageEngineCoreProc
   │   │
   │   ├─> OmniARScheduler.schedule()
   │   │
   │   ├─> ModelRunner.execute_model()
   │   │   - forward pass
   │   │   - KV cache 生成
   │   │
   │   ├─> KV transfer 触发 (prefill_finished)
   │   │   - OmniKVTransferManager.extract_kv()
   │   │   - connector.put() -> Stage 1
   │   │
   │   └─> EngineCoreOutput -> ZMQ PUSH
   │
   └─> Orchestrator 接收输出
       - OutputProcessor.process_outputs()
       - 检测 stage finished

2. Stage 间路由  [Orchestrator 线程]
   │
   ├─> _route_output() 检测 stage_id < final_stage_id
   │
   ├─> _forward_to_next_stage()
   │   │
   │   ├─> stage_client.set_engine_outputs([output])
   │   │
   │   ├─> next_client.process_engine_inputs()
   │   │   - 转换输出为下阶段输入
   │   │
   │   └─> next_client.add_request_async()
   │       - 提交到 Stage 1

3. Stage 1 (Diffusion 模型)
   │
   ├─> StageDiffusionClient.add_request_async()
   │   - kv_sender_info 包含 Stage 0 的地址
   │
   ├─> ZMQ -> StageDiffusionProc
   │   │
   │   ├─> DiffusionEngine.step()
   │   │   │
   │   │   ├─> 接收 KV cache
   │   │   │   - connector.get(from_stage=0)
   │   │   │
   │   │   ├─> DiffusionModelRunner.execute_model()
   │   │   │
   │   │   └─> 生成图像/音频
   │   │
   │   └─> 返回 OmniRequestOutput
   │
   └─> Orchestrator 接收
       - final_output: True
       - 发送到 output_queue

4. 用户获取最终结果
   │
   └─> try_get_output() -> OmniRequestOutput
       - images/audio
       - finished: True
```

---

## 组件生命周期

### AsyncOmniEngine 生命周期

```python
# 1. 创建
engine = AsyncOmniEngine(model, engine_args, ...)

# 内部流程:
__init__():
    # 解析配置
    config_path, stage_configs = _resolve_stage_configs(model, kwargs)

    # 启动 Orchestrator 线程
    self.orchestrator_thread = threading.Thread(
        target=self._bootstrap_orchestrator,
        args=(stage_init_timeout, startup_future),
    )
    self.orchestrator_thread.start()

    # 等待就绪
    self._wait_for_orchestrator_init(startup_future, startup_timeout)

# 2. 运行时
engine.add_request(request_id, prompt, sampling_params_list)
output = engine.try_get_output()
engine.abort([request_id])
result = engine.collective_rpc("profile", args=(True, "trace"))

# 3. 关闭
engine.shutdown()
# 内部:
#   - request_queue.sync_q.put({"type": "shutdown"})
#   - orchestrator_thread.join(timeout=10)
#   - janus queues close
```

### Orchestrator 生命周期

```python
# 在 _bootstrap_orchestrator 中:

def _bootstrap_orchestrator(stage_init_timeout, startup_future):
    # 1. 创建 asyncio 事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # 2. 初始化 janus 队列
    self._initialize_janus_queues()

    # 3. 初始化阶段
    self._initialize_stages(stage_init_timeout)
    # - 启动 LLM 子进程
    # - 启动 Diffusion 子进程
    # - 创建 StageClients

    # 4. 创建 Orchestrator
    orchestrator = Orchestrator(
        request_async_queue=self.request_queue.async_q,
        output_async_queue=self.output_queue.async_q,
        stage_clients=self.stage_clients,
        ...
    )

    # 5. 发送就绪信号
    startup_future.set_result(loop)

    # 6. 运行事件循环
    loop.run_until_complete(orchestrator.run())

    # 7. 清理
    # - 取消待处理任务
    # - 关闭异步生成器
    # - 关闭循环
```

### StageEngineCoreProc 生命周期

```python
# 启动
def spawn_stage_core(vllm_config, executor_class, ...):
    # 1. 分配地址
    addresses = get_engine_zmq_addresses(vllm_config)

    # 2. 创建子进程
    proc = mp.Process(target=StageEngineCoreProc.run_stage_core, ...)
    proc.start()

    return addresses, proc, handshake_address

# 子进程入口
def run_stage_core(vllm_config, ...):
    # 3. 创建 EngineCore
    engine_core = StageEngineCoreProc(vllm_config, ...)

    # 4. HELLO 握手
    handshake_socket.send(msgspec.msgpack.encode({"status": "HELLO"}))

    # 5. 接收 INIT
    init_payload = handshake_socket.recv()
    # 设置 ZMQ 地址

    # 6. READY 握手
    handshake_socket.send(msgspec.msgpack.encode({
        "status": "READY",
        "num_gpu_blocks": num_gpu_blocks,
    }))

    # 7. 运行忙循环
    engine_core.run_busy_loop()
    while True:
        # 处理请求
        # 执行调度
        # 运行模型
        # 发送输出

# 关闭
# 收到 SIGTERM/SIGINT -> 设置 shutdown_state -> 退出循环
```

### DiffusionWorker 生命周期

```python
# WorkerProc.worker_main
def worker_main(rank, od_config, pipe_writer, broadcast_handle, ...):
    # 1. 创建 WorkerProc
    worker_proc = WorkerProc(od_config, gpu_id=rank, ...)

    # 2. 创建 DiffusionWorker
    worker = DiffusionWorker(
        local_rank=rank,
        rank=rank,
        od_config=od_config,
    )
    # - init_device()
    # - DiffusionModelRunner 加载模型
    # - DiffusionLoRAManager 初始化

    # 3. 发送就绪信号
    pipe_writer.send({
        "status": "ready",
        "result_handle": result_mq_handle,
    })

    # 4. 运行忙循环
    worker_proc.worker_busy_loop()
    while _running:
        msg = mq.dequeue(timeout=1.0)
        # 处理 RPC
        # 执行模型
        # 返回结果

# 关闭
# 收到 shutdown 消息 -> _running = False -> shutdown()
```

---

## 总结

vLLM-Omni 通过以下设计实现了多阶段全方位模型的高效服务：

1. **进程隔离**: 每个 LLM/Diffusion 阶段运行在独立子进程中，避免 GIL 竞争，支持独立 GPU 设备分配

2. **分层调度**:
   - Orchestrator 层：跨阶段协调
   - Scheduler 层：每阶段内部调度（OmniARScheduler / DiffusionScheduler）

3. **高效 IPC**:
   - ZMQ：进程间控制流
   - 共享内存：大数据传输
   - Janus：线程间异步桥接

4. **KV Transfer**: 阶段间 KV cache 高效传递，支持 RDMA 等高速传输

5. **灵活部署**:
   - 单进程 inline 模式（单 Diffusion 阶段）
   - 多进程模式（复杂流水线）
   - 单阶段 CLI 模式（分布式推理）
