# Stage Config 配置流程图

## 1. 配置层级结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Mammothmoda2Config (顶层)                             │
│  model_type: "mammothmoda2"                                                  │
│  is_composition: True                                                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    llm_config: Mammothmoda2Qwen2_5_VLConfig         │    │
│  │                                                                      │    │
│  │  ┌───────────────────────────────┐  ┌────────────────────────────┐ │    │
│  │  │    text_config                │  │    vision_config           │ │    │
│  │  │    (文本模型配置)              │  │    (视觉编码器配置)         │ │    │
│  │  │                               │  │                            │ │    │
│  │  │  - hidden_size: 8192          │  │  - hidden_size: 3584       │ │    │
│  │  │  - num_hidden_layers: 80      │  │  - num_heads: 16           │ │    │
│  │  │  - num_attention_heads: 64    │  │  - patch_size: 14          │ │    │
│  │  │  - intermediate_size: 29568   │  │  - depth: 32               │ │    │
│  │  │  - vocab_size: 152064+32800   │  │                            │ │    │
│  │  │  - extra_gen_vocab: True      │  │                            │ │    │
│  │  │  - gen_vocab_size: 32800      │  │                            │ │    │
│  │  └───────────────────────────────┘  └────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │   gen_vae_config (dict)     │  │   gen_dit_config (dict)             │  │
│  │   VAE 解码器配置             │  │   DiT Transformer 配置              │  │
│  │   (diffusers 格式)          │  │   (diffusers 格式)                  │  │
│  └─────────────────────────────┘  └─────────────────────────────────────┘  │
│                                                                              │
│  gen_axes_dim_rope: [40, 40, 40]                                            │
│  gen_axes_lens: [10000, 10000, 10000]                                       │
│  gen_condition_mode: "image"                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Stage 初始化流程

```
用户代码: Omni(model="...", stage_configs_path="...")
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AsyncOmniEngine.__init__()                          │
│                                                                              │
│  ① _resolve_stage_configs()                                                 │
│     └── 加载 YAML → 解析 stage_args → 合并 CLI 参数                          │
│                                                                              │
│  ② 创建通信队列                                                              │
│     └── self.request_queue = janus.Queue()                                  │
│     └── self.output_queue = janus.Queue()                                   │
│                                                                              │
│  ③ 启动 Orchestrator 后台线程                                               │
│     └── threading.Thread(target=self._bootstrap_orchestrator)               │
│                                                                              │
│  ④ 等待初始化完成                                                            │
│     └── startup_future.result(timeout=startup_timeout)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Orchestrator 线程 (asyncio event loop)                    │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║                    _initialize_stages()                                ║ │
│  ║                                                                        ║ │
│  ║  Phase 1: compute_replica_layout()                                     ║ │
│  ║     └── 计算每个 stage 的副本数量和设备分配                              ║ │
│  ║                                                                        ║ │
│  ║  Phase 2: _build_logical_stage_init_plans()                            ║ │
│  ║     ┌──────────────────────────────────────────────────────────────┐  ║ │
│  ║     │  for each stage_config:                                       │  ║ │
│  ║     │      │                                                        │  ║ │
│  ║     │      ├─ extract_stage_metadata() → StageMetadata             │  ║ │
│  ║     │      │                                                        │  ║ │
│  ║     │      ├─ build_engine_args_dict() → dict[str, Any]            │  ║ │
│  ║     │      │                                                        │  ║ │
│  ║     │      └─ build_vllm_config()                                   │  ║ │
│  ║     │            │                                                  │  ║ │
│  ║     │            ├─ OmniEngineArgs(**engine_args_dict)              │  ║ │
│  ║     │            │      │                                           │  ║ │
│  ║     │            │      └─ create_model_config()                    │  ║ │
│  ║     │            │            │                                     │  ║ │
│  ║     │            │            └─ AutoConfig.from_pretrained()       │  ║ │
│  ║     │            │                    │                             │  ║ │
│  ║     │            │                    └─ Mammothmoda2Config 实例    │  ║ │
│  ║     │            │                                                  │  ║ │
│  ║     │            └─ VllmConfig(model_config, cache_config, ...)     │  ║ │
│  ║     └──────────────────────────────────────────────────────────────┘  ║ │
│  ║                                                                        ║ │
│  ║  Phase 3: _initialize_stage_replicas()                                ║ │
│  ║     ┌──────────────────────────────────────────────────────────────┐  ║ │
│  ║     │  for each replica in parallel (ThreadPoolExecutor):          │  ║ │
│  ║     │      │                                                        │  ║ │
│  ║     │      ├─ acquire_device_locks() → 获取 GPU 文件锁              │  ║ │
│  ║     │      │                                                        │  ║ │
│  ║     │      ├─ spawn_stage_core() → 启动子进程                       │  ║ │
│  ║     │      │      │                                                 │  ║ │
│  ║     │      │      └─ multiprocessing.Process(                      │  ║ │
│  ║     │      │             target=StageEngineCoreProc.run_stage_core │  ║ │
│  ║     │      │         )                                              │  ║ │
│  ║     │      │                                                        │  ║ │
│  ║     │      ├─ complete_stage_handshake() → HELLO/INIT/READY 握手    │  ║ │
│  ║     │      │                                                        │  ║ │
│  ║     │      └─ 创建 StageEngineCoreClient                            │  ║ │
│  ║     └──────────────────────────────────────────────────────────────┘  ║ │
│  ║                                                                        ║ │
│  ║  Phase 4: _assemble_stage_pools()                                     ║ │
│  ║     └── 组装 StagePools 供 Orchestrator 调度                          ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║                    Orchestrator.run()                                  ║ │
│  ║                                                                        ║ │
│  ║     while running:                                                     ║ │
│  ║         1. 从 request_queue 获取请求                                   ║ │
│  ║         2. 调度到合适的 StagePool                                      ║ │
│  ║         3. 收集输出结果                                                ║ │
│  ║         4. 发送到 output_queue                                         ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 配置传播链（详细版）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           mammoth_moda2.yaml                                 │
│                                                                              │
│  stage_args:                                                                 │
│    - stage_id: 0                                                            │
│      engine_args:                                                           │
│        model_stage: ar                                                      │
│        model_arch: MammothModa2ForConditionalGeneration                     │
│        ...                                                                  │
│    - stage_id: 1                                                            │
│      engine_args:                                                           │
│        model_stage: dit                                                     │
│        ...                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ load_stage_configs_from_yaml()
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          list[StageConfig]                                   │
│                                                                              │
│  StageConfig(                                                                │
│      stage_id=0,                                                             │
│      engine_args={"model_stage": "ar", ...},                                │
│      runtime=RuntimeConfig(devices="0"),                                    │
│      metadata=StageMetadata(...)                                            │
│  )                                                                          │
│  StageConfig(                                                                │
│      stage_id=1,                                                             │
│      engine_args={"model_stage": "dit", ...},                               │
│      ...                                                                    │
│  )                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ build_vllm_config()
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            VllmConfig                                        │
│                                                                              │
│  VllmConfig(                                                                 │
│      model_config=OmniModelConfig(                                          │
│          hf_config=Mammothmoda2Config(...),                                 │
│          model_stage="ar",  # 或 "dit"                                      │
│          ...                                                                │
│      ),                                                                     │
│      cache_config=CacheConfig(...),                                         │
│      ...                                                                    │
│  )                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ 传递到模型初始化
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              MammothModa2ForConditionalGeneration.__init__()                 │
│                                                                              │
│  cfg = vllm_config.model_config.hf_config  # Mammothmoda2Config             │
│  model_stage = vllm_config.model_config.model_stage  # "ar" 或 "dit"        │
│                                                                              │
│  if model_stage == "ar":                                                    │
│      # AR 阶段使用 VL 配置                                                   │
│      self.ar = init_vllm_registered_model(                                  │
│          vllm_config=vllm_config,                                           │
│          hf_config=cfg.llm_config,  # ← Mammothmoda2Qwen2_5_VLConfig        │
│          architectures=["MammothModa2ARForConditionalGeneration"],          │
│      )                                                                      │
│                                                                              │
│  elif model_stage == "dit":                                                 │
│      # DiT 阶段使用顶层配置                                                   │
│      self.dit = init_vllm_registered_model(                                 │
│          vllm_config=vllm_config,                                           │
│          hf_config=cfg,  # ← Mammothmoda2Config (内部读取 gen_dit_config)   │
│          architectures=["MammothModa2DiTPipeline"],                         │
│      )                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4. 三层配置的访问路径

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           访问规则对照表                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  目标属性          正确访问路径                          错误访问路径        │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  hidden_size       cfg.llm_config.text_config.hidden_size                  │
│                     或 hf_config.get_text_config().hidden_size             │
│                    ───────────────────────────────────────────────────────  │
│                     ❌ cfg.llm_config.hidden_size                          │
│                     (Mammothmoda2Qwen2_5_VLConfig 无此属性)                 │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  num_hidden_layers cfg.llm_config.text_config.num_hidden_layers            │
│                    ───────────────────────────────────────────────────────  │
│                     ❌ cfg.llm_config.num_hidden_layers                    │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  vision_config     cfg.llm_config.vision_config                            │
│                     或 cfg.vision_config (property 代理)                    │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  image_token_id    cfg.llm_config.image_token_id                           │
│                     或 cfg.image_token_id (property 代理)                   │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  gen_dit_config    cfg.gen_dit_config (顶层属性)                            │
│                     (dict 类型，diffusers 配置)                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5. 多进程架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                主进程                                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         用户线程                                      │   │
│  │                                                                       │   │
│  │   omni = Omni(...)                                                   │   │
│  │   result = omni.generate(...)                                        │   │
│  │                                                                       │   │
│  │   ↓ 同步调用                                                          │   │
│  │                                                                       │   │
│  │   request_queue.sync_q.put(request)  ───────────────────────────┐   │   │
│  │                                                                   │   │   │
│  │   result = output_queue.sync_q.get()  ◄─────────────────────────┐│   │   │
│  │                                                                   ││   │   │
│  └──────────────────────────────────────────────────────────────────┼┼───┘   │
│                                                                     ││       │
│  ┌──────────────────────────────────────────────────────────────────┼┼───┐   │
│  │                      Orchestrator 线程                            ││   │   │
│  │                                                                   ││   │   │
│  │   asyncio event loop:                                             ││   │   │
│  │                                                                   ││   │   │
│  │   request = await request_queue.async_q.get()  ◄─────────────────┘│   │   │
│  │                                                                    │   │   │
│  │   stage_pool[stage_id].submit(request)  ──────────────────────────┼─►│   │
│  │                                                                    │   │   │
│  │   result = await stage_pool[stage_id].get_result()  ◄───────────────┘   │
│  │                                                                       │   │
│  │   await output_queue.async_q.put(result)  ─────────────────────────────►│
│  │                                                                       │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ ZMQ
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│ StageEngineCoreProc #0  │ │ StageEngineCoreProc #1  │ │ DiffusionWorker #0      │
│                         │ │                         │ │                         │
│ Stage 0 (AR)            │ │ Stage 1 (DiT)           │ │ Stage N (可选)          │
│                         │ │                         │ │                         │
│ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │
│ │     EngineCore      │ │ │ │     EngineCore      │ │ │ │   DiffusionEngine   │ │
│ │                     │ │ │ │                     │ │ │ │                     │ │
│ │  - model_executor   │ │ │ │  - model_executor   │ │ │ │  - diffusion_model  │ │
│ │  - scheduler        │ │ │ │  - scheduler        │ │ │ │  - scheduler        │ │
│ │                     │ │ │ │                     │ │ │ │                     │ │
│ │  busy_loop():       │ │ │ │  busy_loop():       │ │ │ │  busy_loop():       │ │
│ │    while running:   │ │ │ │    while running:   │ │ │ │    while running:   │ │
│ │      step()         │ │ │ │      step()         │ │ │ │      step()         │ │
│ └─────────────────────┘ │ │ └─────────────────────┘ │ │ └─────────────────────┘ │
│                         │ │                         │ │                         │
│ GPU 0                   │ │ GPU 1                   │ │ GPU 0                   │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
```

## 6. 握手协议序列图

```
主进程                                          子进程 (StageEngineCoreProc)
   │                                                  │
   │                    fork()                         │
   │ ─────────────────────────────────────────────►   │
   │                                                  │
   │                                                  │ 创建 EngineCore
   │                                                  │ 绑定 ZMQ sockets
   │                                                  │
   │◄─────────────────── HELLO ──────────────────────│
   │  {"status": "ready_for_init"}                   │
   │                                                  │
   │                                                  │
   │─────────────── INIT (vllm_config) ─────────────►│
   │                                                  │
   │                                          加载模型权重
   │                                          初始化 KV Cache
   │                                          Profile Run
   │                                                  │
   │◄─────────────────── READY ─────────────────────│
   │  {"status": "ready"}                            │
   │                                                  │
   │                                                  │
   │           进入正常请求处理循环                      │
   │◄──────────────────────────────────────────────►│
   │              SchedulerOutput                     │
   │              EngineCoreOutput                    │
   │                     ...                          │
```

## 7. 配置问题根因图解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         问题一：初始化顺序                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  错误流程：                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  def __init__(self, llm_config, ...):                                │   │
│  │      super().__init__(**kwargs)  ──────────────────────────────┐    │   │
│  │      #                      │                                   │    │   │
│  │      #                      ▼                                   │    │   │
│  │      #              validate_token_ids()                        │    │   │
│  │      #                      │                                   │    │   │
│  │      #                      ▼                                   │    │   │
│  │      #              get_text_config()                           │    │   │
│  │      #                      │                                   │    │   │
│  │      #                      ▼                                   │    │   │
│  │      #              self.llm_config  ──────► AttributeError!    │    │   │
│  │      #                  (未赋值)                                 │    │   │
│  │      self.llm_config = ...  ◄─── 太晚了                         │    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  正确流程：                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  def __init__(self, llm_config, ...):                                │   │
│  │      self.llm_config = ...  ◄─── 先赋值                             │   │
│  │      self.gen_vae_config = ...                                      │   │
│  │      ...                                                            │   │
│  │      super().__init__(**kwargs)  ──────────────────────────────┐    │   │
│  │      #                      │                                   │    │   │
│  │      #                      ▼                                   │    │   │
│  │      #              validate_token_ids()                        │    │   │
│  │      #                      │                                   │    │   │
│  │      #                      ▼                                   │    │   │
│  │      #              get_text_config()                           │    │   │
│  │      #                      │                                   │    │   │
│  │      #                      ▼                                   │    │   │
│  │      #              self.llm_config.text_config ──────► ✓       │    │   │
│  │      #                  (已赋值)                                 │    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      问题二：get_text_config() 返回层级                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  vLLM 期望：                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  get_text_config() ──────► 返回包含以下属性的对象：                    │   │
│  │                              - hidden_size                          │   │
│  │                              - num_attention_heads                  │   │
│  │                              - num_hidden_layers                    │   │
│  │                              - vocab_size                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  错误返回：                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  def get_text_config(self):                                          │   │
│  │      return self.llm_config  # Mammothmoda2Qwen2_5_VLConfig         │   │
│  │                              # ❌ 无 hidden_size 等属性               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  正确返回：                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  def get_text_config(self):                                          │   │
│  │      return self.llm_config.text_config                             │   │
│  │                              # ✓ Mammothmoda2Qwen2_5_VLTextConfig    │   │
│  │                              # ✓ 有 hidden_size 等属性               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      问题三：属性访问路径                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  配置层级：                                                                   │
│                                                                              │
│  Mammothmoda2Config                                                         │
│       │                                                                      │
│       └── llm_config: Mammothmoda2Qwen2_5_VLConfig                         │
│              │                     │                                         │
│              │                     ├── hidden_size  ❌ 不存在               │
│              │                     │                                         │
│              │                     └── num_attention_heads  ❌ 不存在        │
│              │                                                               │
│              └── text_config: Mammothmoda2Qwen2_5_VLTextConfig              │
│                            │                                                 │
│                            ├── hidden_size  ✓ 8192                          │
│                            │                                                 │
│                            └── num_attention_heads  ✓ 64                    │
│                                                                              │
│  错误访问：                                                                   │
│  config.llm_config.hidden_size  ──────► AttributeError                      │
│                                                                              │
│  正确访问：                                                                   │
│  config.llm_config.text_config.hidden_size  ──────► 8192                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
