# vLLM-Omni 架构详解

## 概述

vLLM-Omni 是一个多阶段（multi-stage）的全方位（omnimodal）模型运行时，它在 vLLM V1 架构基础上扩展，支持包含多个阶段的流水线（例如：Thinker AR 模型 -> Talker/TTS -> Diffusion），每个阶段可以是 LLM（自回归）模型或 Diffusion 模型。

---

## 核心组件详解

### 1. AsyncOmniEngine

**文件位置:** `vllm_omni/engine/async_omni_engine.py`

**角色:** 运行在调用者主线程中的轻量级代理。

**进程/线程:** 与调用者同一进程、同一线程。

**职责:**
- 解析 YAML 格式的阶段配置文件
- 启动后台线程运行 Orchestrator
- 通过 **janus 队列**（线程安全的同步/异步桥接）与 Orchestrator 通信
- 提供公共 API：`add_request()`, `abort()`, `collective_rpc()`, `try_get_output()`

**关键属性:**
```python
request_queue: janus.Queue      # 同步队列，向 Orchestrator 发送请求
output_queue: janus.Queue       # 同步队列，从 Orchestrator 接收输出
rpc_output_queue: janus.Queue   # RPC 响应队列
orchestrator_thread: threading.Thread  # 后台线程
```

---

### 2. Orchestrator

**文件位置:** `vllm_omni/engine/orchestrator.py`

**角色:** 中央协调器，运行在专用后台线程中，拥有独立的 asyncio 事件循环。

**进程/线程:** 与 AsyncOmniEngine 同一进程，但运行在**独立线程**中，拥有自己的 asyncio 事件循环。

**职责:**
- 持有所有阶段客户端（包括 LLM 和 Diffusion）
- 持有所有输出处理器
- 在阶段之间路由输出（stage-to-stage transfer 逻辑）
- 管理每个请求的状态（`OrchestratorRequestState`）
- 处理 CFG（Classifier-Free Guidance）伴随请求追踪
- 支持 PD（Prefill-Decode） disaggregation

**关键组件:**
```python
stage_clients: list[Any]                          # StageEngineCoreClient 或 StageDiffusionClient 列表
output_processors: list[MultimodalOutputProcessor]
request_states: dict[str, OrchestratorRequestState]
```

---

### 3. StageEngineCoreClient (LLM 阶段客户端)

**文件位置:** `vllm_omni/engine/stage_engine_core_client.py`

**角色:** LLM/AR 阶段的客户端。继承自 vLLM 的 `AsyncMPClient`。

**进程架构:**
- 客户端运行在 **Orchestrator 线程**中（与 Orchestrator 同进程）
- 实际的 Engine Core 运行在**独立子进程**中（`StageEngineCoreProc`）
- 通过 **ZMQ sockets** 通信（PUSH/PULL 用于输入/输出）

**通信模式:**
```
[Orchestrator 线程]                    [子进程]
StageEngineCoreClient ──ZMQ PUSH──> StageEngineCoreProc
                      <──ZMQ PULL──    (EngineCore busy loop)
```

**关键类:**
- `StageEngineCoreClient` - 标准单引擎客户端
- `DPLBStageEngineCoreClient` - 数据并行负载均衡客户端

---

### 4. StageEngineCoreProc (LLM Engine Core 子进程)

**文件位置:** `vllm_omni/engine/stage_engine_core_proc.py`

**角色:** 运行 LLM EngineCore 忙循环的子进程。继承自 vLLM 的 `EngineCoreProc`。

**进程:** 运行在**独立子进程**中，通过 `multiprocessing.Process` 启动。

**生命周期:**
1. 通过 `spawn_stage_core()` 启动
2. 通过 ZMQ DEALER/ROUTER 进行 HELLO 握手
3. 发送 INIT 携带引擎地址
4. READY 握手，携带 num_gpu_blocks
5. 运行 `run_busy_loop()` 处理请求

---

### 5. StageDiffusionClient (Diffusion 阶段客户端 - 进程外)

**文件位置:** `vllm_omni/diffusion/stage_diffusion_client.py`

**角色:** Diffusion 阶段的客户端。启动并与 `StageDiffusionProc` 通信。

**进程架构:**
- 客户端运行在 **Orchestrator 线程**中
- Diffusion 引擎运行在**独立子进程**中（`StageDiffusionProc`）
- 通过 **ZMQ sockets** 通信（PUSH/PULL）

**通信模式:**
```
[Orchestrator 线程]                    [子进程]
StageDiffusionClient ──ZMQ PUSH──> StageDiffusionProc
                     <──ZMQ PULL──   (DiffusionEngine)
```

**关键方法:**
- `add_request_async()` - 提交 diffusion 请求
- `add_batch_request_async()` - 提交批量提示
- `get_diffusion_output_nowait()` - 非阻塞输出检索
- `collective_rpc_async()` - 转发控制 RPC 到子进程

---

### 6. InlineStageDiffusionClient (Diffusion 阶段客户端 - 进程内)

**文件位置:** `vllm_omni/diffusion/inline_stage_diffusion_client.py`

**角色:** 替代客户端，使用 ThreadPoolExecutor 在进程内运行 DiffusionEngine。

**使用场景:** 当只有一个 Diffusion 阶段时使用（避免子进程开销）。

**进程架构:**
- 完全运行在 **Orchestrator 进程**中
- 使用 `ThreadPoolExecutor`（1 个 worker）处理阻塞的 diffusion 操作
- 通过 `asyncio.Queue` 通信（无 ZMQ 开销）

---

### 7. StageDiffusionProc (Diffusion 子进程)

**文件位置:** `vllm_omni/diffusion/stage_diffusion_proc.py`

**角色:** Diffusion 推理的子进程入口点。

**进程:** 运行在**独立子进程**中。

**生命周期:**
1. 初始化 DiffusionEngine
2. 通过握手 socket 发送 READY
3. 运行异步事件循环（`run_loop()`）处理 ZMQ 消息

**关键操作:**
- 处理 `add_request`, `add_batch_request`, `abort`, `collective_rpc` 消息
- 使用 `ThreadPoolExecutor` 处理阻塞的 `DiffusionEngine.step()` 调用

---

### 8. DiffusionEngine

**文件位置:** `vllm_omni/diffusion/diffusion_engine.py`

**角色:** 主要的 Diffusion 推理引擎。

**组件:**
```python
executor: DiffusionExecutor   # 管理工作进程
scheduler: SchedulerInterface  # 请求调度器
pipeline                     # 实际的 diffusion 模型（通过 DiffusersPipelineLoader 加载）
```

**进程位置:**
- 运行在 `StageDiffusionProc` 子进程内，或
- 运行在 `InlineStageDiffusionClient` 内（进程内，通过线程池）

---

### 9. MultiprocDiffusionExecutor

**文件位置:** `vllm_omni/diffusion/executor/multiproc_executor.py`

**角色:** 管理多个 Diffusion 工作进程。

**进程架构:**
```
[MultiprocDiffusionExecutor]              [Worker 进程]
      (在 DiffusionEngine)                     (DiffusionWorker)
            │                                       │
            └── MessageQueue (broadcast) ─────────>│
            <── MessageQueue (results) ────────────┘
```

**通信机制:**
- 使用 vLLM 的 `MessageQueue`（共享内存广播）
- 广播请求到所有 workers
- 仅 rank-0 发送结果返回

---

### 10. DiffusionWorker

**文件位置:** `vllm_omni/diffusion/worker/diffusion_worker.py`

**角色:** Diffusion 模型的 GPU 工作进程。

**进程:** 运行在**独立子进程**中（每个 GPU 一个）。

**组件:**
- `DiffusionModelRunner` - 处理模型加载和执行
- `DiffusionLoRAManager` - 可选的 LoRA 支持
- Sleep/wake 支持用于内存管理

---

### 11. DiffusionModelRunner

**文件位置:** `vllm_omni/diffusion/worker/diffusion_model_runner.py`

**角色:** 处理模型加载、编译、缓存和执行。

**关键特性:**
- 加载 diffusion 模型（diffusers 格式）
- 可选的 torch.compile 支持
- 缓存加速（Cache-DiT, TeaCache）
- CPU offloading 支持
- 多阶段流水线的 KV transfer

---

### 12. WorkerProc

**文件位置:** `vllm_omni/diffusion/worker/diffusion_worker.py`

**角色:** 在独立进程中运行一个 Worker 的包装器。

**进程:** 运行在**独立子进程**中。

**职责:**
- 管理与主进程的 IPC
- 运行 worker busy loop
- 处理 RPC 请求
- 管理 sleep/wake 任务

---

### 13. OutputProcessor

**文件位置:** `vllm_omni/engine/output_processor.py`

**角色:** 将原始引擎输出处理为面向用户的 `RequestOutput` 对象。

**进程/线程:** 运行在 **Orchestrator 线程**中。

---

### 14. OmniMasterServer

**文件位置:** `vllm_omni/engine/stage_engine_startup.py`

**角色:** 单阶段 CLI 模式下的协调服务器，用于跨进程阶段注册和地址发现。

**进程:** 运行在**主进程**中。

---

## 进程/线程/协程组织架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           主进程 (Caller 进程)                                │
│                                                                             │
│  ┌─────────────────────┐                                                   │
│  │  AsyncOmniEngine    │ (主线程)                                          │
│  │    - janus queues   │                                                   │
│  │    - 公共 API        │                                                   │
│  └──────────┬──────────┘                                                   │
│             │ spawn thread                                                  │
│             ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Orchestrator (后台线程, 独立 asyncio 事件循环)                        │   │
│  │    - stage_clients                                                  │   │
│  │    - output_processors                                              │   │
│  │    - request_states                                                 │   │
│  │    - asyncio 任务: _request_handler, _orchestration_output_handler  │   │
│  └──────────┬──────────────────────────────────────────────────────────┘   │
│             │                                                               │
│  ┌──────────▼──────────────────────────────────────────────────────────┐   │
│  │  StageEngineCoreClient (LLM) 或 StageDiffusionClient                │   │
│  │    (在 Orchestrator 线程)                                           │   │
│  └──────────┬─────────────────────────────────┬────────────────────────┘   │
│             │ ZMQ                            │ ZMQ                         │
└─────────────┼────────────────────────────────┼─────────────────────────────┘
              │                                │
      ┌───────▼───────┐                ┌───────▼───────┐
      │StageEngineCore│                │StageDiffusion │
      │     Proc      │                │     Proc      │
      │  (子进程)      │                │  (子进程)      │
      └───────┬───────┘                └───────┬───────┘
              │                                │
      ┌───────▼───────┐                ┌───────▼───────┐
      │  LLM Workers  │                │ Diffusion     │
      │  (vLLM GPU    │                │ Workers       │
      │   workers)    │                │ (Multiproc)   │
      └───────────────┘                └───────┬───────┘
                                               │
                                       ┌───────▼───────┐
                                       │ WorkerProc    │
                                       │  (子进程 x N)  │
                                       │ DiffusionModel│
                                       │   Runner      │
                                       └───────────────┘
```

---

## 通信机制详解

### 1. Janus Queues（线程到线程）

**用于:** AsyncOmniEngine（主线程）<-> Orchestrator（后台线程）

**类型:** `janus.Queue` - 线程安全的同步/异步桥接

**消息类型:**
- `add_request` - 添加请求
- `streaming_update` - 流式更新
- `abort` - 中止请求
- `collective_rpc` - RPC 调用
- `shutdown` - 关闭

**代码示例:**
```python
# AsyncOmniEngine 中（同步端）
self.request_queue.sync_q.put_nowait(msg)
result = self.output_queue.sync_q.get(timeout=timeout)

# Orchestrator 中（异步端）
msg = await self.request_async_queue.get()
await self.output_async_queue.put(output_msg)
```

---

### 2. ZMQ Sockets（进程到进程）

**用于:**
- StageEngineCoreClient <-> StageEngineCoreProc（LLM 阶段）
- StageDiffusionClient <-> StageDiffusionProc（Diffusion 阶段）

**Socket 类型:**
- PUSH/PULL - 用于请求/响应通道
- ROUTER/DEALER - 用于握手

**代码示例:**
```python
# StageDiffusionClient
self._request_socket = self._zmq_ctx.socket(zmq.PUSH)
self._request_socket.connect(request_address)
self._response_socket = self._zmq_ctx.socket(zmq.PULL)
self._response_socket.connect(response_address)

# 发送请求
self._request_socket.send(encoder.encode({"type": "add_request", ...}))

# 接收响应
raw = self._response_socket.recv(zmq.NOBLOCK)
```

---

### 3. MessageQueue（共享内存广播）

**用于:** MultiprocDiffusionExecutor <-> DiffusionWorker 进程

**库:** vLLM 的 `shm_broadcast.MessageQueue`

**特性:**
- 高效广播到多个 workers
- POSIX 共享内存用于大型张量

**代码示例:**
```python
# MultiprocDiffusionExecutor
self._broadcast_mq = MessageQueue(
    n_reader=num_workers,
    n_local_reader=num_workers,
    local_reader_ranks=list(range(num_workers)),
)
self._broadcast_mq.enqueue(rpc_request)
response = self._result_mq.dequeue()

# WorkerProc
self.mq = MessageQueue.create_from_handle(broadcast_handle, gpu_id)
msg = self.mq.dequeue(timeout=1.0)
```

---

### 4. POSIX 共享内存

**用于:** Diffusion workers 与 executor 之间传输大型张量

**实现:** `diffusion/ipc.py`

**特性:**
- 大于 1MB 的张量复制到命名共享内存段
- 仅轻量级元数据通过 MessageQueue 发送

---

### 5. KV Transfer（阶段间）

**用于:** 阶段之间传输 KV cache（例如：Thinker -> Diffusion）

**实现:** `distributed/omni_connectors/kv_transfer_manager.py`

**机制:**
- 分布式设置的 ZMQ 传输
- RDMA 的 Mooncake transfer engine connector

---

## 请求流程

### 单阶段 Diffusion

```
1. 用户调用 AsyncOmniEngine.add_request()
2. 消息通过 janus 队列排队到 Orchestrator
3. Orchestrator 接收消息，路由到 StageDiffusionClient
4. StageDiffusionClient 通过 ZMQ 发送到 StageDiffusionProc
5. StageDiffusionProc 分发到 DiffusionEngine.step()
6. DiffusionEngine 通过 MultiprocDiffusionExecutor 运行
7. Workers 通过 DiffusionModelRunner 处理
8. 结果通过链路返回
```

### 多阶段流水线（Thinker -> Talker -> Diffusion）

```
1. 请求提交到 Stage 0 (Thinker)
2. Thinker 生成 tokens
3. Orchestrator 检测阶段完成
4. Orchestrator 调用 _forward_to_next_stage()
5. Stage-0 输出由 stage-1 输入处理器处理
6. 请求提交到 Stage 1 (Talker)
7. ... 每个阶段继续
8. 最终阶段输出返回给用户
```

---

## 关键设计模式

### 1. 关注点分离

每个组件有单一职责：
- **AsyncOmniEngine**: API 层
- **Orchestrator**: 协调
- **Stage Clients**: 每阶段通信
- **Workers**: 模型执行

### 2. 子进程隔离

繁重计算运行在独立进程中：
- 防止 GIL 竞争
- 支持每阶段内存隔离
- 允许独立 GPU 设备分配

### 3. 分层抽象

vLLM 模式扩展到多阶段：
- `StageEngineCoreClient` 扩展 `AsyncMPClient`
- `StageEngineCoreProc` 扩展 `EngineCoreProc`

### 4. 灵活部署

- 单进程模式：用于单个 diffusion 阶段
- 多进程模式：用于复杂流水线
- 单阶段 CLI 模式：用于分布式推理

---

## 配置示例

### 多阶段配置（YAML）

```yaml
stages:
  - stage_id: 0
    stage_type: llm
    model_stage: thinker
    engine_args:
      model: "model_name"
      tensor_parallel_size: 2
    final_output: false

  - stage_id: 1
    stage_type: llm
    model_stage: talker
    engine_args:
      model: "model_name"
    final_output: false

  - stage_id: 2
    stage_type: diffusion
    engine_args:
      model: "model_name"
      tensor_parallel_size: 2
    final_output: true
    final_output_type: audio
```

---

## 总结

vLLM-Omni 的架构设计通过进程隔离、分层抽象和灵活的通信机制，实现了：

1. **多阶段流水线支持** - 可组合 LLM 和 Diffusion 阶段
2. **高性能 IPC** - ZMQ + 共享内存减少序列化开销
3. **独立扩展** - 每个阶段可独立配置并行度和设备
4. **向后兼容** - 复用 vLLM V1 的 EngineCore 架构
5. **灵活部署** - 支持单进程、多进程和分布式模式
