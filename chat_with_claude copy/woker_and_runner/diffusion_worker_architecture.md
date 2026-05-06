# Diffusion Worker 架构解析

## 概览

`diffusion_worker.py` 中包含三个核心类，它们构成了一个层次化的工作器架构：

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WorkerProc                                   │
│  (进程级包装器 - 管理IPC通信和消息循环)                               │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    WorkerWrapperBase                           │  │
│  │  (动态扩展包装器 - 支持扩展类和委托机制)                        │  │
│  │                                                                │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │                  DiffusionWorker                         │  │  │
│  │  │  (核心工作器 - GPU基础设施和模型执行)                     │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 一、DiffusionWorker（核心工作器）

### 1.1 职责定位

**核心职责**：管理 GPU 基础设施，委托模型操作给 `DiffusionModelRunner`

```python
class DiffusionWorker:
    """
    A worker that manages GPU infrastructure and delegates to the model runner.

    This class handles infrastructure initialization only:
    - Device setup (CUDA device selection)
    - Distributed environment (NCCL, model parallel)
    - Memory management (sleep/wake)

    All model-related operations (loading, compilation, execution) are
    delegated to DiffusionModelRunner.
    """
```

### 1.2 核心属性

```python
def __init__(self, local_rank, rank, od_config, skip_load_model=False):
    self.local_rank = local_rank          # 本地 GPU 序号
    self.rank = rank                      # 全局 rank
    self.od_config = od_config            # OmniDiffusionConfig 配置
    self.device: torch.device             # CUDA 设备
    self.vllm_config: VllmConfig          # vLLM 配置
    self.model_runner: DiffusionModelRunner  # 模型运行器
    self.lora_manager: DiffusionLoRAManager  # LoRA 管理器
    self.profiler: WorkerProfiler         # 性能分析器
```

### 1.3 初始化流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    DiffusionWorker.__init__()                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. init_device()                                                │
│     ├── 设置分布式环境变量 (MASTER_ADDR, RANK, etc.)             │
│     ├── 获取并设置 CUDA 设备                                     │
│     ├── 创建 VllmConfig                                          │
│     └── 初始化分布式环境 (NCCL, Model Parallel)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 创建 DiffusionModelRunner                                    │
│     └── 传入 vllm_config, od_config, device                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. load_model() [如果 skip_load_model=False]                   │
│     ├── 调用 model_runner.load_model()                           │
│     └── 同步 GPU，触发垃圾回收                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. init_lora_manager() [如果 skip_load_model=False]            │
│     └── 创建 DiffusionLoRAManager                                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 核心方法

| 方法 | 职责 | 说明 |
|------|------|------|
| `init_device()` | 设备初始化 | 设置 CUDA 设备、分布式环境、模型并行 |
| `load_model()` | 模型加载 | 委托给 `model_runner.load_model()` |
| `execute_model()` | 模型执行 | 委托给 `model_runner.execute_model()` |
| `sleep()` | 休眠模式 | 卸载模型权重，释放 GPU 内存 |
| `wake_up()` | 唤醒模式 | 重新加载模型权重到 GPU |
| `handle_sleep_task()` | 处理休眠任务 | 完整的休眠任务处理流程 |
| `handle_wake_task()` | 处理唤醒任务 | 完整的唤醒任务处理流程 |

---

## 二、WorkerWrapperBase（动态扩展包装器）

### 2.1 职责定位

**核心职责**：创建 `DiffusionWorker` 实例，支持动态继承扩展类

```python
class WorkerWrapperBase:
    """
    Wrapper base class that creates DiffusionWorker with optional
    worker_extension_cls support. This enables dynamic inheritance
    for DiffusionWorker to extend with custom functionality.
    """
```

### 2.2 扩展机制

```
┌─────────────────────────────────────────────────────────────────┐
│              WorkerWrapperBase._prepare_worker_class()           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────────┐                   ┌───────────────────────┐
│ 无扩展类          │                   │ 有扩展类               │
│                   │                   │                       │
│ 返回 DiffusionWorker                │ 动态创建新类:          │
│                   │                   │   DiffusionWorkerWith │
│                   │                   │   + <ExtensionClass>  │
└───────────────────┘                   └───────────────────────┘
```

### 2.3 动态继承实现

```python
def _prepare_worker_class(self) -> type:
    worker_class = self.base_worker_class  # 默认是 DiffusionWorker

    if self.worker_extension_cls:
        # 解析扩展类
        worker_extension_cls = resolve_obj_by_qualname(self.worker_extension_cls)

        # 动态创建新类
        class_name = f"{worker_class.__name__}With{worker_extension_cls.__name__}"
        worker_class = type(class_name, (worker_extension_cls, worker_class), {})

    return worker_class
```

**示例**：当传入 `worker_extension_cls=CustomPipelineWorkerExtension` 时：

```python
# 动态生成的新类
class DiffusionWorkerWithCustomPipelineWorkerExtension(CustomPipelineWorkerExtension, DiffusionWorker):
    pass

# 新类拥有两个父类的方法
# - CustomPipelineWorkerExtension.re_init_pipeline()
# - DiffusionWorker.load_model(), execute_model(), etc.
```

### 2.4 方法委托

```python
def __getattr__(self, attr: str):
    """Delegate attribute access to the wrapped worker."""
    return getattr(self.worker, attr)

def execute_method(self, method: str, *args, **kwargs):
    """Execute a method on the worker."""
    func = getattr(self.worker, method)
    return func(*args, **kwargs)
```

---

## 三、WorkerProc（进程级包装器）

### 3.1 职责定位

**核心职责**：在独立进程中运行 Worker，管理 IPC 通信和消息循环

```python
class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""
```

### 3.2 核心属性

```python
def __init__(self, od_config, gpu_id, broadcast_handle, wake_event, ...):
    self.od_config = od_config            # OmniDiffusionConfig
    self.gpu_id = gpu_id                  # GPU ID
    self.wake_event = wake_event          # 唤醒事件（跨进程同步）

    # IPC 通信
    self.context = zmq.Context()          # ZMQ 上下文
    self.mq = MessageQueue()              # 输入消息队列
    self.result_mq = MessageQueue()       # 输出消息队列（仅 rank 0）

    # Worker 实例
    self.worker = self._create_worker()   # WorkerWrapperBase 实例
    self._running = True                  # 运行状态标志
```

### 3.3 进程启动流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    主进程                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. 创建 mp.Event (wake_event)                            │   │
│  │ 2. 创建 MessageQueue (broadcast_handle)                  │   │
│  │ 3. mp.Process(target=WorkerProc.worker_main, args=...)  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ fork()
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Worker 子进程                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ WorkerProc.worker_main():                                │   │
│  │                                                          │   │
│  │ 1. 加载插件 (load_omni_general_plugins)                  │   │
│  │ 2. 创建 WorkerProc 实例                                  │   │
│  │    └── 内部创建 WorkerWrapperBase                        │   │
│  │        └── 内部创建 DiffusionWorker                      │   │
│  │ 3. 通过 pipe 发送 "ready" 状态                           │   │
│  │ 4. 进入 worker_busy_loop() 消息循环                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 消息循环

```
┌─────────────────────────────────────────────────────────────────┐
│                    WorkerProc.worker_busy_loop()                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────┴────────────────────┐
         │                                         │
         ▼                                         ▼
┌─────────────────┐                     ┌─────────────────────┐
│ mq.dequeue()    │                     │ wake_event.is_set() │
│ timeout=1.0s    │                     │ (OOB POKE 检查)     │
└────────┬────────┘                     └──────────┬──────────┘
         │                                         │
         │ 无消息                                   │ 被设置
         │                                         │
         ▼                                         ▼
    继续循环                            ┌─────────────────────┐
                                       │ 构造 wake_up 消息   │
                                       │ msg = {"type":      │
                                       │   "wake_up", ...}   │
                                       └──────────┬──────────┘
                                                  │
         ┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      消息类型分发                                │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─── msg["type"] == "sleep"
         │         │
         │         ▼
         │    ┌─────────────────────────────────┐
         │    │ worker.handle_sleep_task()      │
         │    │ return_result(ack)              │
         │    └─────────────────────────────────┘
         │
         ├─── msg["type"] == "wake_up"
         │         │
         │         ▼
         │    ┌─────────────────────────────────┐
         │    │ worker.handle_wake_task()       │
         │    │ return_result(ack)              │
         │    └─────────────────────────────────┘
         │
         ├─── msg["type"] == "rpc"
         │         │
         │         ▼
         │    ┌─────────────────────────────────┐
         │    │ execute_rpc(msg)                │
         │    │ return_result(result)           │
         │    └─────────────────────────────────┘
         │
         ├─── msg["type"] == "shutdown"
         │         │
         │         ▼
         │    ┌─────────────────────────────────┐
         │    │ _running = False                │
         │    │ 退出循环                        │
         │    └─────────────────────────────────┘
         │
         └─── 其他（生成请求）
                   │
                   ▼
              ┌─────────────────────────────────┐
              │ worker.execute_model(msg,       │
              │              od_config)         │
              │ return_result(output)           │
              └─────────────────────────────────┘
```

### 3.5 IPC 通信架构

```
┌──────────────────────────────────────────────────────────────────┐
│                          主进程                                   │
│                                                                  │
│  ┌──────────────────┐        ┌──────────────────┐               │
│  │ Scheduler/Server │        │   Result Queue   │               │
│  │                  │        │    Reader        │               │
│  └────────┬─────────┘        └────────▲─────────┘               │
│           │                           │                         │
│           │ enqueue(msg)              │ dequeue()               │
│           │                           │                         │
└───────────┼───────────────────────────┼─────────────────────────┘
            │                           │
            │                           │
            │  Shared Memory            │  Shared Memory
            │  MessageQueue             │  MessageQueue
            │                           │
┌───────────┼───────────────────────────┼─────────────────────────┐
│           │                           │                         │
│           ▼                           │                         │
│  ┌──────────────────┐                 │                         │
│  │ WorkerProc       │                 │                         │
│  │                  │        ┌────────┴────────┐                │
│  │  mq.dequeue()────┘        │ result_mq       │                │
│  │                           │ .enqueue()──────┘                │
│  └──────────────────────────────────────────────────────────────┘
│                          Worker 子进程                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 四、完整生命周期

```
时间轴 ──────────────────────────────────────────────────────────────►

主进程:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 创建 IPC    │───►│ 启动 Worker │───►│ 发送请求    │
│ 资源        │    │ 子进程      │    │             │
└─────────────┘    └─────────────┘    └──────┬──────┘
      │                  │                   │
      │                  │                   │
      ▼                  ▼                   ▼
子进程:           ┌─────────────┐    ┌─────────────┐
                  │ worker_main │───►│ busy_loop   │
                  │ 初始化      │    │ 处理请求    │
                  └─────────────┘    └─────────────┘
                        │
                        ▼
                  ┌─────────────────────────────────────────────┐
                  │ WorkerProc                                   │
                  │   └── WorkerWrapperBase                      │
                  │         └── DiffusionWorker                  │
                  │               └── DiffusionModelRunner       │
                  │                     └── Pipeline             │
                  └─────────────────────────────────────────────┘
```

---

## 五、请求执行流程

### 5.1 生成请求

```
Scheduler
    │
    │ request = OmniDiffusionRequest(...)
    │ mq.enqueue(request)
    ▼
WorkerProc.worker_busy_loop()
    │
    │ msg = mq.dequeue()
    ▼
worker.execute_model(msg, od_config)
    │
    │ WorkerWrapperBase.__getattr__("execute_model")
    │     │
    │     └──► DiffusionWorker.execute_model()
    │               │
    │               └──► DiffusionModelRunner.execute_model()
    │                         │
    │                         └──► Pipeline.__call__()
    ▼
DiffusionOutput
    │
    │ result_mq.enqueue(output)
    ▼
Scheduler 收到结果
```

### 5.2 RPC 请求

```
主进程发送 RPC:
    {
        "type": "rpc",
        "method": "load_weights",
        "args": [...],
        "kwargs": {...},
        "output_rank": 0
    }

WorkerProc.execute_rpc()
    │
    │ should_execute = (output_rank == self.gpu_id)
    │ should_reply = (output_rank == self.gpu_id) and (self.result_mq is not None)
    ▼
worker.execute_method(method, *args, **kwargs)
    │
    │ WorkerWrapperBase.execute_method()
    │     │
    │     └──► getattr(self.worker, method)(*args, **kwargs)
    ▼
返回结果给主进程
```

### 5.3 Sleep/Wake 请求

```
主进程发送 Sleep:
    {"type": "sleep", "level": 2, "task_id": "xxx"}

WorkerProc.worker_busy_loop()
    │
    │ msg["type"] == "sleep"
    ▼
worker.handle_sleep_task(task)
    │
    │ DiffusionWorker.handle_sleep_task()
    │     │
    │     ├── sleep(level=task.level)
    │     │     └── CuMemAllocator.sleep()
    │     │
    │     ├── all_reduce(freed_bytes)  # 多 GPU 同步
    │     │
    │     └── 返回 OmniACK (仅 rank 0)
    ▼
return_result(ack)
    │
    │ result_mq.enqueue(ack)
    ▼
主进程收到 ACK
```

---

## 六、三者的核心区别

| 维度 | DiffusionWorker | WorkerWrapperBase | WorkerProc |
|------|-----------------|-------------------|------------|
| **层次** | 核心实现层 | 中间包装层 | 进程管理层 |
| **职责** | GPU 基础设施、模型执行 | 动态扩展、方法委托 | IPC 通信、消息循环 |
| **运行位置** | 子进程 | 子进程 | 子进程 |
| **生命周期** | 被 Wrapper 持有 | 被 WorkerProc 持有 | 独立进程主循环 |
| **扩展性** | 可被继承扩展 | 提供扩展机制 | 固定逻辑 |
| **通信** | 无直接 IPC | 无直接 IPC | 管理 ZMQ/MessageQueue |

---

## 七、设计模式

### 7.1 装饰器模式（Wrapper Pattern）

```
WorkerWrapperBase 是装饰器：
- 持有 DiffusionWorker 实例
- 可以动态添加扩展功能
- 通过 __getattr__ 委托方法调用
```

### 7.2 策略模式（Strategy Pattern）

```
worker_extension_cls 是策略：
- 通过配置注入不同的扩展类
- 运行时动态选择行为
- 例如：CustomPipelineWorkerExtension
```

### 7.3 进程模式（Process Pattern）

```
WorkerProc 管理进程生命周期：
- 在独立进程中运行
- 管理进程间通信
- 处理进程启动和关闭
```

---

## 八、关键设计决策

### 8.1 为什么需要三层结构？

1. **DiffusionWorker**: 专注于 GPU 和模型逻辑，不涉及进程和 IPC
2. **WorkerWrapperBase**: 提供扩展机制，支持自定义功能注入
3. **WorkerProc**: 管理进程级事务，隔离 IPC 复杂性

### 8.2 为什么使用动态继承？

```python
# 问题：需要在运行时为 DiffusionWorker 添加新功能
# 方案：动态创建子类

# 传统方式（不灵活）：
class ExtendedDiffusionWorker(CustomPipelineWorkerExtension, DiffusionWorker):
    pass

# 动态继承方式（灵活）：
worker_class = type(
    "DiffusionWorkerWithCustomPipelineWorkerExtension",
    (CustomPipelineWorkerExtension, DiffusionWorker),
    {}
)
```

好处：
- 运行时可配置扩展类
- 支持多种扩展组合
- 无需修改 DiffusionWorker 源码

### 8.3 为什么 WorkerProc 管理消息循环？

```python
# WorkerProc 是进程入口点
def worker_main(rank, od_config, ...):
    worker_proc = WorkerProc(...)
    worker_proc.worker_busy_loop()  # 主循环在这里
```

原因：
- 分离关注点：进程管理 vs 模型执行
- WorkerProc 处理 IPC、消息分发
- DiffusionWorker 专注于模型逻辑

---

## 九、总结

```
┌─────────────────────────────────────────────────────────────────┐
│                          架构总览                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WorkerProc (进程级)                                            │
│  ├── 管理 ZMQ、MessageQueue (IPC)                               │
│  ├── 运行消息循环 (worker_busy_loop)                            │
│  └── 持有 WorkerWrapperBase                                     │
│                                                                 │
│      WorkerWrapperBase (包装层)                                 │
│      ├── 动态继承扩展类 (worker_extension_cls)                  │
│      ├── 方法委托 (__getattr__)                                 │
│      └── 持有 DiffusionWorker                                   │
│                                                                 │
│          DiffusionWorker (核心层)                               │
│          ├── 管理 GPU 设备、分布式环境                          │
│          ├── 委托模型操作给 DiffusionModelRunner                │
│          ├── 管理 LoRA、Profiler                                │
│          └── 处理 sleep/wake 逻辑                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

这种三层架构实现了：
- **关注点分离**：进程管理、扩展机制、模型执行各司其职
- **可扩展性**：通过 WorkerWrapperBase 支持动态功能注入
- **进程隔离**：WorkerProc 在独立进程中运行，不阻塞主进程
- **灵活通信**：通过 MessageQueue 实现高效 IPC
