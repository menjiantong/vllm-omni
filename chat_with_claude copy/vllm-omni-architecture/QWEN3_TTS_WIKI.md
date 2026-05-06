# Qwen3-TTS End-to-End Inference Wiki

本文档详细讲解 Qwen3-TTS 模型在 vLLM-Omni 框架中的端到端推理流程，涵盖模型架构、核心组件、调用链路、矩阵维度等完整技术细节。

---

## 目录

1. [概述](#1-概述)
2. [模型架构](#2-模型架构)
3. [核心组件详解](#3-核心组件详解)
4. [完整调用链路](#4-完整调用链路)
5. [调度系统](#5-调度系统)
6. [Prefill与Decode阶段](#6-prefill与decode阶段)
7. [码本与音频编码](#7-码本与音频编码)
8. [高维空间词传输](#8-高维空间词传输)
9. [Thinker与Talker组件](#9-thinker与talker组件)
10. [自定义算子与CUDA Graph](#10-自定义算子与cuda-graph)
11. [任务类型详解](#11-任务类型详解)
12. [矩阵形状与维度汇总](#12-矩阵形状与维度汇总)

---

## 1. 概述

### 1.1 什么是 Qwen3-TTS

Qwen3-TTS 是阿里巴巴通义千问团队开发的端到端文本转语音（TTS）模型，具有以下特点：

- **两阶段架构**：Talker（文本→音频码本）+ Code2Wav（码本→波形）
- **多任务支持**：CustomVoice（预置音色）、VoiceDesign（声音设计）、Base（声音克隆）
- **高质量音频输出**：24kHz 采样率，16层 RVQ 码本
- **流式推理支持**：支持实时音频流输出

### 1.2 vLLM-Omni 框架

vLLM-Omni 是 vLLM 的多模态扩展框架，专门用于支持非自回归结构和非文本输出的模型：

```
┌─────────────────────────────────────────────────────────────────┐
│                        vLLM-Omni 架构                            │
├─────────────────────────────────────────────────────────────────┤
│  Omni / AsyncOmni (用户入口)                                     │
│       ↓                                                          │
│  AsyncOmniEngine (引擎核心)                                      │
│       ↓                                                          │
│  Orchestrator (多阶段协调器，后台线程运行)                         │
│       ↓                                                          │
│  StageEngineCoreClient[] (各阶段客户端)                          │
│       ↓                                                          │
│  ModelRunner (GPUARModelRunner / GPUGenerationModelRunner)      │
│       ↓                                                          │
│  Model (Qwen3TTSTalker / Qwen3TTSCode2Wav)                      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 代码入口文件结构

`end2end.py` 文件的核心流程：

```
main() / main_streaming()
    ↓
_build_inputs()           # 构建输入数据
    ↓
Omni.from_cli_args()      # 初始化引擎
    ↓
omni.generate()           # 执行推理
    ↓
_save_wav()               # 保存音频输出
```

---

## 2. 模型架构

### 2.1 两阶段流水线

Qwen3-TTS 采用两阶段流水线架构：

```
┌──────────────────┐      ┌──────────────────┐
│   Stage 0        │      │   Stage 1        │
│   Talker         │ ───→ │   Code2Wav       │
│   (LLM_AR)       │      │   (LLM_GENERATION)│
│                  │      │                  │
│  文本 → 码本      │      │  码本 → 波形      │
│  [T, Q] codes    │      │  [wav_len] audio │
└──────────────────┘      └──────────────────┘
```

**流水线配置定义** (`vllm_omni/model_executor/models/qwen3_tts/pipeline.py`):

```python
QWEN3_TTS_PIPELINE = PipelineConfig(
    model_type="qwen3_tts",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="qwen3_tts",              # Talker 阶段
            execution_type=StageExecutionType.LLM_AR,  # 自回归
            engine_output_type="latent",          # 输出隐向量（码本）
            sampling_constraints={"stop_token_ids": [2150]},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",               # Code2Wav 阶段
            execution_type=StageExecutionType.LLM_GENERATION,  # 生成式
            final_output=True,
            final_output_type="audio",            # 最终输出音频
        ),
    ),
)
```

### 2.2 配置层次结构

```
Qwen3TTSConfig (顶层配置)
├── talker_config: Qwen3TTSTalkerConfig
│   ├── hidden_size: 1024          # Talker 隐藏层维度
│   ├── num_hidden_layers: 20      # Transformer 层数
│   ├── num_attention_heads: 16    # 注意力头数
│   ├── num_code_groups: 32        # RVQ 码本组数
│   ├── text_hidden_size: 2048     # 文本嵌入维度
│   └── code_predictor_config: Qwen3TTSTalkerCodePredictorConfig
│       ├── hidden_size: 1024      # Code Predictor 隐藏维度
│       ├── num_hidden_layers: 5   # Code Predictor 层数
│       └── num_code_groups: 32    # 码本组数
└── speaker_encoder_config: Qwen3TTSSpeakerEncoderConfig
    ├── enc_dim: 1024              # 说话人嵌入维度
    └── mel_dim: 128               # Mel 频谱维度
```

### 2.3 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_code_groups` | 32 | RVQ 码本层数（量化器数量） |
| `vocab_size` | 2048 | 每层码本的词汇表大小 |
| `codec_eos_token_id` | 4198 | 码本序列结束标记 |
| `codec_bos_id` | 4197 | 码本序列开始标记 |
| `codec_pad_id` | 4196 | 码本填充标记 |
| `hidden_size` | 1024 | Talker 隐藏状态维度 |
| `text_hidden_size` | 2048 | 文本嵌入维度（来自文本模型） |

---

## 3. 核心组件详解

### 3.1 Talker 模型 (Stage 0)

**文件**: `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py`

**类**: `Qwen3TTSTalkerForConditionalGeneration`

**作用**: 将文本转换为 RVQ 离散音频码本。

**核心组件**:

```
Qwen3TTSTalkerForConditionalGeneration
├── model: Qwen3Model                    # 主 Transformer 骨干
│   ├── embed_tokens: ParallelLMHead     # 词嵌入层
│   ├── layers: ModuleList[Qwen3DecoderLayer]  # 20层解码器
│   └── norm: RMSNorm                    # 层归一化
├── lm_head: ParallelLMHead              # 输出投影到词汇表
├── text_embedding: nn.Embedding         # 文本 token 嵌入
├── text_projection: Qwen3TTSTalkerResizeMLP  # 文本嵌入维度映射
├── speaker_encoder: Qwen3TTSSpeakerEncoder   # ECAPA-TDNN 说话人编码器
└── code_predictor: CodePredictorWrapper      # 残差码本预测器
```

#### 3.1.1 Speaker Encoder (ECAPA-TDNN)

**作用**: 从参考音频提取说话人身份向量（1024维）。

**架构**:

```
Qwen3TTSSpeakerEncoder
├── blocks: ModuleList                   # TDNN + SERes2Net 块
│   ├── TimeDelayNetBlock               # 时延神经网络
│   └── SqueezeExcitationRes2NetBlock   # SE-Res2Net 块
├── mfa: TimeDelayNetBlock              # 多层特征聚合
├── asp: AttentiveStatisticsPooling     # 注意力统计池化
└── fc: Conv1d                          # 最终投影

输入: Mel 频谱图 [B, T, 128]
输出: 说话人嵌入 [B, 1024]
```

**ECAPA-TDNN 细节**:

- **TimeDelayNetBlock**: 1D 卷积 + ReLU 激活
- **Res2NetBlock**: 残差分层网络，scale=8
- **SqueezeExcitationBlock**: 通道注意力机制
- **AttentiveStatisticsPooling**: 自注意力统计池化

#### 3.1.2 Code Predictor

**文件**: `vllm_omni/model_executor/models/common/qwen3_code_predictor.py`

**作用**: 从第0层码本预测剩余 Q-1 层码本（无需 KV Cache 的重新 Prefill）。

**架构**:

```
CodePredictorWrapper
├── model: CodePredictorBaseModel
│   ├── codec_embedding: ModuleList[nn.Embedding]  # Q-1 个嵌入表
│   ├── layers: ModuleList[CodePredictorDecoderLayer]  # 5层解码器
│   └── rotary_emb: _RotaryEmbedding              # 旋转位置编码
└── lm_head: ModuleList[nn.Linear]               # 每码本组一个输出头

输入: layer0_code [B], layer0_embed [B, H], last_talker_hidden [B, H]
输出: all_codes [B, Q]  (Q=32)
```

**工作流程**:

1. 填充 buffer 位置 0（Talker 隐藏状态）和位置 1（第0层嵌入）
2. 对于 step in 1..Q-1:
   - 运行 Transformer forward
   - 采样下一个码本值（top-k + top-p 或 argmax）
   - 嵌入预测的码本值 → 下一个 buffer 位置
3. 返回所有码本 [B, Q]

### 3.2 Code2Wav 模型 (Stage 1)

**文件**: `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py`

**类**: `Qwen3TTSCode2Wav`

**作用**: 将 RVQ 离散码本解码为音频波形。

**核心组件**:

```
Qwen3TTSCode2Wav
├── _speech_tokenizer: Qwen3TTSTokenizer  # 音频编解码器包装
├── _decoder: nn.Module                    # SpeechTokenizer 解码器
├── _num_quantizers: int                   # 16 (12Hz) / 8 (25Hz)
├── _output_sample_rate: int               # 24000 Hz
└── _total_upsample: int                   # 480 (每帧波形采样数)
```

**Forward 流程**:

```python
def forward(self, input_ids, **kwargs):
    # input_ids: 扁平化码本 [q*F]
    # q = num_quantizers, F = num_frames

    codes_qf = flat.reshape(q, frames)  # [Q, F]
    codes_bqf = codes_qf.unsqueeze(0)   # [1, Q, F]

    # 通过解码器解码
    wav = decoder.chunked_decode(codes_bqf)  # [1, 1, wav_len]

    # 裁剪上下文前缀
    wav = wav[start:end]

    return OmniOutput(
        multimodal_outputs={"model_outputs": wav, "sr": sr}
    )
```

### 3.3 Audio Tokenizer (Speech Tokenizer)

**文件**: `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_tokenizer.py`

**两个版本**:

| 版本 | 帧率 | 码本层数 | 特点 |
|------|------|----------|------|
| 12Hz (V2) | 12 fps | 16 | 更简洁的架构 |
| 25Hz (V1) | 25 fps | 8 | 需要 x-vector 和 ref_mels |

**关键方法**:

```python
class Qwen3TTSTokenizer:
    def encode(audio, sr) -> audio_codes:  # 音频 → 码本
        # 返回 [T, Q] 形状的离散码

    def decode(audio_codes) -> audio:       # 码本 → 音频
        # 返回波形数组
```

**RVQ 结构**:

```
音频码本形状: [T, Q]
- T = 帧数 (duration * frame_rate)
- Q = 量化器数量 (通常为 16)

每帧产生 Q 个离散码:
  Layer 0 (语义层): 捕获音素/语言内容
  Layers 1-15 (声学层): 捕获精细声学细节
```

---

## 4. 完整调用链路

### 4.1 初始化阶段

```
main()
    ↓
Omni.from_cli_args(args, model=model_name)
    ↓
AsyncOmniEngine.__init__()
    ├── _resolve_stage_configs()           # 加载流水线拓扑
    │       └── QWEN3_TTS_PIPELINE        # 从注册表获取
    ├── _bootstrap_orchestrator()          # 启动协调器线程
    │       └── _initialize_stages()       # 初始化各阶段
    │               ├── _launch_llm_stage(stage_0)  # Talker
    │               └── _launch_llm_stage(stage_1)  # Code2Wav
    └── _wait_for_orchestrator_init()      # 等待初始化完成
```

### 4.2 请求提交流程

```
omni.generate(batch)
    ↓
for prompt in inputs:
    ↓
engine.add_request(request)
    ├── InputProcessor.process_inputs()    # 分词、多模态预处理
    ├── _upgrade_to_omni_request()         # 注入 additional_information
    └── output_processors[0].add_request() # 注册输出处理

    ↓ (通过 janus Queue 传递到协调器线程)

Orchestrator._handle_add_request()
    ├── stage_clients[0].add_request_async()  # 提交到 Stage 0
    └── request_states[request_id] = state    # 记录请求状态
```

### 4.3 Stage 0 执行流程 (Talker)

```
StageEngineCoreClient.add_request_async()
    ↓
GPUARModelRunner.execute_model()
    ├── _update_states(scheduler_output)   # 更新批次状态
    ├── _preprocess()                       # 输入预处理
    │       └── model.preprocess()          # 模型特定预处理
    ├── _model_forward()                    # Transformer 前向传播
    │       └── Qwen3Model.forward()
    │               ├── self_attn()         # 自注意力
    │               └── mlp()               # MLP 层
    └── extract_multimodal_outputs()        # 提取多模态输出

    ↓ (返回 None，等待 sample_tokens)

GPUARModelRunner.sample_tokens()
    ├── _sample(logits)                     # 采样下一个 token
    │       └── model.sample() 或 sampler() # 自定义或标准采样器
    ├── _talker_mtp_forward()               # Code Predictor 前向
    │       └── code_predictor(layer0_code, ...)
    │               └── 返回 audio_codes [B, Q]
    └── 构建 OmniModelRunnerOutput
            ├── sampled_token_ids
            ├── pooler_output = {"audio_codes": tensor, "hidden": tensor}
            └── multimodal_outputs
```

### 4.4 Stage 间传输流程

```
Orchestrator._orchestration_loop()
    ↓ (轮询各阶段输出)
Orchestrator._process_stage_outputs(stage_id=0, raw_outputs)
    ↓
Orchestrator._route_output()
    ├── output.finished == False → 继续生成
    └── output.finished == True → _forward_to_next_stage()

    ↓
Orchestrator._forward_to_next_stage()
    ├── stage_clients[0].set_engine_outputs()  # 设置当前阶段输出
    └── next_client.process_engine_inputs()    # 处理下一阶段输入

    ↓
talker2code2wav() 或 talker2code2wav_async_chunk()  # 阶段输入处理器
    ├── 提取 audio_codes [T, Q]
    ├── 过滤无效帧（超出码本范围、零填充）
    ├── 重塑为 codebook-major: [Q, F] → [q*F]
    └── 构建 OmniTokensPrompt

    ↓
StageEngineCoreClient.add_request_async()  # 提交到 Stage 1
```

### 4.5 Stage 1 执行流程 (Code2Wav)

```
GPUGenerationModelRunner.execute_model()
    ├── _update_states()
    ├── _preprocess()
    └── _run_generation_model()
            └── Qwen3TTSCode2Wav.forward()
                    ├── _ensure_speech_tokenizer_loaded()  # 懒加载解码器
                    ├── _split_request_ids()               # 按请求分割
                    ├── codes_qf = flat.reshape(q, frames) # 重塑码本
                    ├── decoder.chunked_decode(codes_bqf)  # 解码波形
                    └── 返回 OmniOutput(audio=wav, sr=24000)

    ↓
GPUGenerationModelRunner.sample_tokens()
    └── 构建 pooler_output = [{"model_outputs": wav, "sr": sr}]
```

### 4.6 输出返回流程

```
Orchestrator._orchestration_loop()
    ↓ (检测到最终阶段输出)
output_async_queue.put(output)
    ↓ (回到主线程)
AsyncOmniEngine.try_get_output()
    ↓
OmniRequestOutput
    ├── request_id
    ├── finished: True
    ├── final_output_type: "audio"
    └── outputs[0].multimodal_output = {"audio": wav, "sr": 24000}

    ↓
_save_wav()
    └── sf.write(out_wav, audio_tensor, samplerate=24000)
```

---

## 5. 调度系统

### 5.1 调度器类型

vLLM-Omni 为不同执行类型提供专用调度器：

| 调度器 | 执行类型 | 用途 |
|--------|----------|------|
| `OmniARScheduler` | LLM_AR | 自回归阶段（Talker） |
| `OmniGenerationScheduler` | LLM_GENERATION | 生成式阶段（Code2Wav） |
| `OmniDiffusionScheduler` | DIFFUSION | 扩散模型 |

### 5.2 OmniARScheduler

**文件**: `vllm_omni/core/sched/omni_ar_scheduler.py`

**扩展自**: vLLM 的 `Scheduler`

**关键功能**:

```python
class OmniARScheduler(Scheduler):
    def schedule(self):
        # 1. 处理待处理的音频块（流式模式）
        chunk_transfer_adapter.process_pending_chunks()

        # 2. 调用基础调度
        scheduler_output = super().schedule()

        # 3. 包装请求数据
        return OmniSchedulerOutput(
            new_request_data=OmniNewRequestData(...),
            ...
        )

    def update_from_output(self, output):
        # 1. 预处理 KV 提取确认
        # 2. 更新请求状态
        # 3. 检查 KV 传输触发条件
        # 4. 处理失败的 KV 加载
```

**KV 传输触发条件**:

1. `prefill_finished`: Prefill 完成时触发
2. `special_token`: 特定 token 生成时触发

### 5.3 OmniGenerationScheduler

**文件**: `vllm_omni/core/sched/omni_generation_scheduler.py`

**特点**:

- 一次性调度（所有 token 同时处理）
- 支持 async_chunk 模式通过 `OmniChunkTransferAdapter`
- 无需 token 采样循环

### 5.4 调度流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                      调度器工作流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  请求队列                                                        │
│  ┌───┐ ┌───┐ ┌───┐                                             │
│  │ R1│ │ R2│ │ R3│                                             │
│  └───┘ └───┘ └───┘                                             │
│     ↓                                                           │
│  ┌─────────────────────────────────────┐                       │
│  │      Scheduler.schedule()           │                       │
│  │  - 检查内存限制                      │                       │
│  │  - 选择可调度的请求                  │                       │
│  │  - 构建 SchedulerOutput             │                       │
│  └─────────────────────────────────────┘                       │
│     ↓                                                           │
│  ┌─────────────────────────────────────┐                       │
│  │   GPUModelRunner.execute_model()    │                       │
│  │  - 前向传播                         │                       │
│  │  - 返回 hidden_states               │                       │
│  └─────────────────────────────────────┘                       │
│     ↓                                                           │
│  ┌─────────────────────────────────────┐                       │
│  │   GPUModelRunner.sample_tokens()    │                       │
│  │  - 采样 token                       │                       │
│  │  - 构建 ModelRunnerOutput           │                       │
│  └─────────────────────────────────────┘                       │
│     ↓                                                           │
│  ┌─────────────────────────────────────┐                       │
│  │  Scheduler.update_from_output()     │                       │
│  │  - 更新请求状态                      │                       │
│  │  - 处理完成请求                      │                       │
│  └─────────────────────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Prefill与Decode阶段

### 6.1 概念区分

| 阶段 | 说明 | Token 数量 |
|------|------|-----------|
| **Prefill** | 处理完整 prompt，填充 KV Cache | N (prompt 长度) |
| **Decode** | 逐个生成新 token | 1 每步 |

### 6.2 Talker 的 Prefill 流程

```
Prefill 阶段 (首次前向传播)
    ↓
输入: prompt_token_ids (完整文本 + 特殊标记)
    │   形状: [prompt_len]
    │
    ├── 嵌入层: input_ids → inputs_embeds
    │   形状: [prompt_len, hidden_size=1024]
    │
    ├── Transformer 层 (×20)
    │   ├── Self-Attention
    │   │   ├── Q, K, V 投影
    │   │   ├── 旋转位置编码 (RoPE)
    │   │   ├── 注意力计算: softmax(QK^T/√d)V
    │   │   └── 输出投影
    │   └── MLP
    │       ├── gate_proj: [H, intermediate_size]
    │       ├── up_proj: [H, intermediate_size]
    │       └── down_proj: [intermediate_size, H]
    │
    └── 输出: hidden_states [prompt_len, H]
            存储到 KV Cache
```

### 6.3 Talker 的 Decode 流程

```
Decode 阶段 (迭代生成)
    ↓
每步输入: last_token_id (单个 token)
    │   形状: [1]
    │
    ├── 嵌入: token_id → embed
    │   形状: [1, H]
    │
    ├── 从 KV Cache 加载历史 K, V
    │
    ├── Transformer 层 (×20)
    │   ├── 仅计算新 token 的注意力
    │   └── 更新 KV Cache
    │
    ├── LM Head: hidden → logits
    │   形状: [1, vocab_size=3072]
    │
    ├── 采样: logits → next_token
    │   └── 如果 next_token == stop_token_id (2150):
    │       └── 停止生成
    │
    └── Code Predictor (每步调用)
        ├── 输入: layer0_code, last_talker_hidden
        ├── 预测 layers 1..Q-1
        └── 输出: audio_codes [1, Q]
```

### 6.4 Code2Wav 执行模式

Code2Wav 是 **LLM_GENERATION** 类型，不区分 Prefill/Decode：

```
一次性执行
    ↓
输入: 完整码本序列 [q*F]
    │   q = num_quantizers = 16
    │   F = num_frames
    │
    ├── 重塑: [q*F] → [Q, F]
    │
    ├── RVQ 解码
    │   ├── rvq_first.decode(codes[:, :n_q_semantic])
    │   └── rvq_rest.decode(codes[:, n_q_semantic:])
    │
    ├── Transformer 处理
    │
    ├── 卷积上采样
    │   └── upsample_rates = (8, 5, 4, 3) → total = 480
    │
    └── 输出: 波形 [wav_len]
            wav_len = F * 480 (24kHz)
```

---

## 7. 码本与音频编码

### 7.1 RVQ (Residual Vector Quantization) 原理

RVQ 将连续的音频特征量化为多层离散码本：

```
原始音频特征 x
    ↓
┌─────────────────────────────────────┐
│         Layer 0 (语义层)             │
│  code_0 = argmin ||x - e_0||        │
│  residual_1 = x - e_0[code_0]       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         Layer 1                     │
│  code_1 = argmin ||r_1 - e_1||      │
│  residual_2 = r_1 - e_1[code_1]     │
└─────────────────────────────────────┘
    ↓
    ... (重复 16 层)
    ↓
最终码本: [code_0, code_1, ..., code_15]
```

### 7.2 码本参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_quantizers` | 16 | RVQ 层数（12Hz 版本） |
| `codebook_size` | 2048 | 每层码本的条目数 |
| `encode_downsample_rate` | 1920 | 每码本帧对应波形采样数 |
| `frame_rate` | 12 Hz | 每秒码本帧数 |
| `input_sample_rate` | 16000 Hz | 输入音频采样率 |
| `output_sample_rate` | 24000 Hz | 输出音频采样率 |

### 7.3 码本形状转换

```
时间维度:
  - 音频时长: duration 秒
  - 码本帧数: T = duration * 12 (帧/秒)
  - 波形采样数: N = duration * 24000 (采样/秒)
  - 关系: N = T * 1920 (每帧 1920 个波形采样)

码本表示:
  - 时间主序: [T, Q] = [帧数, 量化器数]
  - 码本主序 (Code2Wav 输入): [Q, F] → 扁平化为 [q*F]

示例 (1秒音频):
  - T = 12 帧
  - Q = 16 量化器
  - audio_codes 形状: [12, 16]
  - 扁平化后: [16 * 12] = [192]
  - 波形采样数: 12 * 1920 = 23040 ≈ 24000
```

### 7.4 码本词汇表

```python
# 特殊标记
codec_eos_token_id = 4198   # 结束标记
codec_bos_id = 4197         # 开始标记
codec_pad_id = 4196         # 填充标记
codec_think_id = 4202       # Think 标记
codec_nothink_id = 4203     # Nothing 标记

# 有效码本值范围
valid_codes = 0..2047       # 每层码本的有效值

# 过滤无效帧
valid_mask = audio_codes.any(dim=1) & (audio_codes.max(dim=1).values < 2048)
```

---

## 8. 高维空间词传输

### 8.1 隐向量传输机制

在多阶段流水线中，阶段间通过 **隐向量 (Hidden States)** 传输高维表示：

```
Stage 0 (Talker) 输出:
    ├── hidden_states: [seq_len, 1024]
    │   └── 用于 Code Predictor 输入
    └── audio_codes: [T, Q]
        └── 用于 Code2Wav 输入

传输路径:
    GPU (Talker) → CPU → GPU (Code2Wav)
```

### 8.2 Pooler Output 结构

```python
pooler_output: list[dict[str, object]] = [
    {
        "hidden": torch.Tensor,       # 隐藏状态 [scheduled_tokens, H]
        "audio_codes": torch.Tensor,  # 音频码本 [T, Q]
        "ref_code": torch.Tensor,     # 参考码本 (Base 模式)
        "ref_code_len": int,          # 参考码本长度
    },
    ...
]
```

### 8.3 阶段输入处理器

**文件**: `vllm_omni/model_executor/stage_input_processors/qwen3_tts.py`

**非流式处理** (`talker2code2wav`):

```python
def talker2code2wav(stage_list, engine_input_source, prompt):
    """收集所有 Talker 码本，一次性传递给 Code2Wav。"""

    for talker_output in talker_outputs:
        audio_codes = output.multimodal_output["audio_codes"]  # [T, Q]

        # 过滤无效帧
        valid_mask = audio_codes.any(dim=1) & (audio_codes.max(dim=1).values < 2048)
        audio_codes = audio_codes[valid_mask]

        # 预置参考码本 (声音克隆)
        if ref_code is not None:
            audio_codes = torch.cat([ref_code, audio_codes], dim=0)

        # 转换为码本主序并扁平化
        codec_codes = audio_codes.transpose(0, 1).reshape(-1).tolist()  # [q*F]

    return [OmniTokensPrompt(prompt_token_ids=codec_codes, ...)]
```

**流式处理** (`talker2code2wav_async_chunk`):

```python
def talker2code2wav_async_chunk(transfer_manager, pooling_output, request, is_finished):
    """每 N 帧传递一次码本块给 Code2Wav。"""

    # 提取最新帧
    frame = _extract_last_frame(pooling_output)  # [Q]
    transfer_manager.code_prompt_token_ids[request_id].append(frame)

    # 配置
    chunk_size = 25              # 每块帧数
    left_context_size = 25       # 左上下文帧数

    length = len(transfer_manager.code_prompt_token_ids[request_id])

    # 每 chunk_size 帧或完成时触发
    if length % chunk_size == 0 or is_finished:
        window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

        # 预置参考码本
        if ref_code is not None:
            window_frames = ref_frames + window_frames

        # 扁平化
        code_predictor_codes = [window_frames[f][q] for q in range(Q) for f in range(num_frames)]

        return {
            "code_predictor_codes": code_predictor_codes,
            "left_context_size": left_context_size,
            "finished": is_finished,
        }
```

### 8.4 共享内存传输

对于大规模隐向量传输，使用共享内存减少拷贝开销：

```yaml
# 部署配置
connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      codec_chunk_frames: 25
      codec_left_context_frames: 25
```

---

## 9. Thinker与Talker组件

### 9.1 命名来源

在 Qwen-Omni 多模态模型中：

- **Thinker**: 负责多模态理解、文本生成的"思考"阶段
- **Talker**: 负责语音合成、音频生成的"表达"阶段

对于纯 TTS 模型 (Qwen3-TTS)，只有 Talker 阶段。

### 9.2 Talker 内部结构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Talker 内部结构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   输入                                                          │
│   ├── text_tokens: 文本 token IDs                               │
│   ├── speaker_embed: 说话人嵌入 (Base 模式)                      │
│   └── ref_code_context: 参考码本上下文 (ICL 模式)                │
│                                                                 │
│   ┌─────────────────────────────────────────┐                   │
│   │         Prompt 构建                      │                   │
│   │  [语言标记] + [说话人标记] + [文本]       │                   │
│   │  + [参考码本] (可选)                     │                   │
│   └─────────────────────────────────────────┘                   │
│                      ↓                                          │
│   ┌─────────────────────────────────────────┐                   │
│   │      Text Embedding + Projection        │                   │
│   │  [text_hidden=2048] → [talker_hidden=1024] │               │
│   └─────────────────────────────────────────┘                   │
│                      ↓                                          │
│   ┌─────────────────────────────────────────┐                   │
│   │      Qwen3Model (Transformer)           │                   │
│   │  20 层解码器，hidden_size=1024           │                   │
│   │  GQA: num_heads=16, kv_heads=2          │                   │
│   └─────────────────────────────────────────┘                   │
│                      ↓                                          │
│   ┌─────────────────────────────────────────┐                   │
│   │         LM Head + 采样                   │                   │
│   │  hidden → logits [vocab=3072]           │                   │
│   │  logits → layer0_code                   │                   │
│   └─────────────────────────────────────────┘                   │
│                      ↓                                          │
│   ┌─────────────────────────────────────────┐                   │
│   │       Code Predictor                    │                   │
│   │  输入: layer0_code, hidden              │                   │
│   │  输出: audio_codes [Q=32]               │                   │
│   │  内部: 5层 Transformer, 无 KV Cache     │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
│   输出                                                          │
│   └── audio_codes: [T, Q] (累积)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 Code Predictor 的"Re-Prefill"策略

与主 Talker 不同，Code Predictor 不维护 KV Cache：

```
每步 Code Predictor 调用:
    ↓
重新构建输入:
    ├── buffer[0] = last_talker_hidden  # 来自主模型
    ├── buffer[1] = layer0_embed        # 第0层码本嵌入
    └── buffer[2..Q] = 预测嵌入 (逐步填充)

    ↓
完整前向传播 (无 KV Cache):
    └── 所有 5 层 Transformer 完整计算

    ↓
预测下一层码本:
    └── lm_head[q] → code_q
```

**设计原因**: Code Predictor 只有 5 层，序列长度固定为 Q (32)，重新计算的代价低于维护 KV Cache 的内存开销。

---

## 10. 自定义算子与CUDA Graph

### 10.1 CUDA Graph 加速原理

CUDA Graph 将 GPU 操作序列捕获为单个可重放图：

```
传统执行:
    CPU → GPU kernel 1 → CPU → GPU kernel 2 → ...
    (每次都有 CPU-GPU 同步开销)

CUDA Graph:
    CPU → 捕获图 → 重放图 (最小化 CPU 开销)
```

### 10.2 Talker MTP CUDA Graph

**文件**: `vllm_omni/worker/gpu_ar_model_runner.py`

```python
def _capture_talker_mtp_graphs(self):
    """为 Code Predictor 捕获 CUDA Graph。"""

    capture_sizes = self.compilation_config.cudagraph_capture_sizes

    for bsz in capture_sizes:
        # 准备固定缓冲区
        ids = self.talker_mtp_input_ids.gpu[:bsz]
        emb = self.talker_mtp_inputs_embeds.gpu[:bsz]
        hid = self.last_talker_hidden.gpu[:bsz]
        ts = self.text_step.gpu[:bsz]

        # 捕获
        with set_forward_context(..., cudagraph_runtime_mode=CUDAGraphMode.FULL):
            self.talker_mtp(ids, emb, hid, ts)
```

### 10.3 Code2Wav CUDA Graph

**文件**: `vllm_omni/model_executor/models/qwen3_tts/cuda_graph_decoder_wrapper.py`

```python
class CUDAGraphDecoderWrapper:
    def chunked_decode_with_cudagraph(self, codes, chunk_size=300, left_context_size=25):
        """分块解码，支持左上下文。"""

        while start_index < total_len:
            # 获取带左上下文的块
            codes_chunk = codes[..., start_index - context_size : end_index]

            # 解码
            wav_chunk = self.decode(codes_chunk)

            # 裁剪上下文部分
            wavs.append(wav_chunk[..., context_size * total_upsample :])

        return torch.cat(wavs, dim=-1)
```

### 10.4 自定义 Triton 算子

音频处理中的自定义算子：

```python
# SnakeBeta 激活函数 (用于音频解码器)
class SnakeBeta(nn.Module):
    def forward(self, x):
        # x = x * sigmoid(beta * x) + alpha * x
        return x * torch.sigmoid(self.beta * x) + self.alpha * x

# 预计算指数缓存
def precompute_snake_caches(self):
    # 为 Triton kernel 预计算
    self.exp_cache = torch.exp(...)
```

---

## 11. 任务类型详解

### 11.1 CustomVoice (预置音色)

**特点**:
- 使用模型预定义的说话人 ID
- 无需参考音频
- 默认非流式模式

**配置**:

```python
additional_information = {
    "task_type": ["CustomVoice"],
    "text": ["要合成的文本"],
    "language": ["Chinese"],     # 语言
    "speaker": ["Vivian"],       # 说话人名称
    "instruct": ["用愤怒的语气说"],  # 可选指令
    "max_new_tokens": [2048],
}
```

**说话人映射** (`talker_config.spk_id`):

```python
spk_id = {
    "Vivian": 0,
    "Ryan": 1,
    ...
}
```

### 11.2 VoiceDesign (声音设计)

**特点**:
- 从文本描述生成声音特征
- 无预定义说话人
- 默认非流式模式

**配置**:

```python
additional_information = {
    "task_type": ["VoiceDesign"],
    "text": ["要合成的文本"],
    "language": ["Chinese"],
    "instruct": ["萝莉女声，音调偏高，撒娇语气"],  # 声音描述
    "max_new_tokens": [2048],
}
```

### 11.3 Base (声音克隆)

**特点**:
- 需要参考音频
- 支持流式模式
- 两种子模式：ICL 和 X-vector Only

**配置**:

```python
additional_information = {
    "task_type": ["Base"],
    "ref_audio": ["path/to/reference.wav"],  # 参考音频
    "ref_text": ["参考音频的文本内容"],        # 参考文本
    "text": ["要合成的文本"],
    "language": ["Auto"],
    "x_vector_only_mode": [False],  # True 为 X-vector Only 模式
    "max_new_tokens": [2048],
}
```

**处理流程**:

```
Base 模式处理:
    ↓
1. 加载参考音频
    └── MediaConnector.fetch_audio(ref_audio)
        └── 返回 (waveform, sample_rate)

    ↓
2. 提取说话人嵌入
    └── SpeakerEncoder(ref_audio)
        ├── Mel 频谱提取
        ├── ECAPA-TDNN 前向
        └── 返回 speaker_embed [1024]

    ↓
3. 编码参考音频 (ICL 模式)
    └── SpeechTokenizer.encode(ref_audio)
        └── 返回 ref_code [T_ref, Q]

    ↓
4. 构建 ICL Prompt
    ├── speaker_embed → prompt embedding
    ├── ref_code → context codes
    └── syn_text → generation target
```

---

## 12. 矩阵形状与维度汇总

### 12.1 Talker 模型

| 张量 | 形状 | 说明 |
|------|------|------|
| `input_ids` | [B, seq_len] | 输入 token IDs |
| `inputs_embeds` | [B, seq_len, 1024] | 输入嵌入 |
| `hidden_states` | [total_tokens, 1024] | 展平的隐藏状态 |
| `logits` | [num_decode_tokens, 3072] | 输出 logits |
| `audio_codes` | [B, Q=32] | 每步预测的码本 |

### 12.2 Code Predictor

| 张量 | 形状 | 说明 |
|------|------|------|
| `layer0_code` | [B] | 第0层码本值 |
| `layer0_embed` | [B, 1024] | 第0层嵌入 |
| `buffer` | [B, Q+1, 1024] | 内部缓冲区 |
| `all_codes` | [B, Q=32] | 所有码本预测 |

### 12.3 Code2Wav 模型

| 张量 | 形状 | 说明 |
|------|------|------|
| `input_ids` | [q*F] | 扁平化码本 |
| `codes_qf` | [Q=16, F] | 重塑后的码本 |
| `codes_bqf` | [1, Q, F] | 批次化的码本 |
| `wav_output` | [wav_len] | 音频波形 |

### 12.4 Speaker Encoder

| 张量 | 形状 | 说明 |
|------|------|------|
| `mels` | [B, T, 128] | Mel 频谱图 |
| `hidden` | [B, 1536, T] | 中间特征 |
| `speaker_embedding` | [B, 1024] | 说话人身份向量 |

### 12.5 批处理维度

```
batch_size = num_requests
seq_len = prompt_length + generated_tokens

# Prefill 阶段
total_tokens = sum(prompt_lengths)  # 所有请求的 prompt 总长度

# Decode 阶段
total_tokens = batch_size           # 每请求一个 token

# CUDA Graph 填充
padded_tokens = next_power_of_2(total_tokens)
```

---

## 附录 A: 关键文件索引

| 文件路径 | 说明 |
|----------|------|
| `vllm_omni/model_executor/models/qwen3_tts/pipeline.py` | 流水线配置 |
| `vllm_omni/model_executor/models/qwen3_tts/configuration_qwen3_tts.py` | 模型配置类 |
| `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py` | Talker 模型 |
| `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_code2wav.py` | Code2Wav 模型 |
| `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_tokenizer.py` | 音频编解码器 |
| `vllm_omni/model_executor/stage_input_processors/qwen3_tts.py` | 阶段输入处理器 |
| `vllm_omni/engine/async_omni_engine.py` | 异步引擎核心 |
| `vllm_omni/engine/orchestrator.py` | 多阶段协调器 |
| `vllm_omni/worker/gpu_ar_model_runner.py` | AR 模型运行器 |
| `vllm_omni/worker/gpu_generation_model_runner.py` | 生成模型运行器 |
| `vllm_omni/core/sched/omni_ar_scheduler.py` | AR 调度器 |

---

## 附录 B: 常见问题排查

### B.1 音频输出静音或失真

**可能原因**:
1. 码本值超出有效范围 (0-2047)
2. 参考码本长度配置错误
3. 左上下文帧数不足

**排查方法**:
```python
# 检查码本范围
assert audio_codes.min() >= 0
assert audio_codes.max() < 2048

# 检查左上下文
left_context_frames = 25  # 推荐 ≥ decoder.sliding_window
```

### B.2 流式输出延迟过高

**可能原因**:
1. 初始块大小过大
2. 网络传输延迟

**优化建议**:
```yaml
# 部署配置优化
extra:
  codec_chunk_frames: 25
  codec_left_context_frames: 25
  initial_codec_chunk_frames: 10  # 更小的初始块
```

### B.3 显存不足

**排查方法**:
```bash
# 查看显存使用
nvidia-smi

# 调整 GPU 显存利用率
gpu_memory_utilization: 0.8  # 降低
```

---

## 附录 C: 性能优化建议

### C.1 CUDA Graph 优化

```python
# 配置捕获大小
compilation_config:
  cudagraph_capture_sizes: [1, 2, 4, 8, 16, 32]
  cudagraph_num_of_warmups: 2
```

### C.2 批处理优化

```python
# 批大小必须是 2 的幂
batch_size = 8  # 推荐

# 启用前缀缓存
enable_prefix_caching: true
```

### C.3 流式优化

```python
# 更大的块大小减少调度开销
codec_chunk_frames: 50

# 更大的左上下文提高质量
codec_left_context_frames: 50
```

---

*文档版本: 1.0*
*最后更新: 2026-04-23*
*适用版本: vLLM-Omni (Qwen3-TTS)*
