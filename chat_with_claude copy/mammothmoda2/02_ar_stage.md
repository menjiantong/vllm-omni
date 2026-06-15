# AR Stage (自回归阶段) 架构详解

## 1. 类层次结构

```
MammothModa2ForConditionalGeneration (顶层入口)
│
├── model_stage == "ar"
│   └── MammothModa2ARForConditionalGeneration
│       │   继承自 Qwen2_5_VLForConditionalGeneration
│       │
│       ├── visual (Vision Tower)
│       │   └── Qwen2_5_VLVisionTransformer
│       │
│       └── language_model (MoE LLM)
│           └── MammothModa2Qwen2ForCausalLM
│               ├── embed_tokens (基础词嵌入)
│               ├── gen_embed_tokens (生成词嵌入, 可选)
│               ├── layers (Decoder Layers)
│               │   └── Mammoth2DecoderLayer × N
│               │       ├── self_attn
│               │       ├── mlp (Understanding Expert)
│               │       └── gen_mlp (Generation Expert, 可选)
│               ├── norm (RMSNorm)
│               ├── lm_head (基础输出头)
│               └── gen_head (生成输出头, 可选)
│
└── model_stage == "dit" → DiT Stage (见另一文档)
```

## 2. 核心组件详解

### 2.1 MammothModa2ForConditionalGeneration

**位置**: `mammoth_moda2.py:694-827`

**职责**: 根据 `model_stage` 分发到对应的子模型

**关键属性**:
```python
self.model_stage: str      # "ar" | "dit" | "vae"
self.ar: nn.Module         # AR 阶段模型
self.dit: nn.Module        # DiT 阶段模型
```

**初始化逻辑**:
```python
if self.model_stage == "ar":
    self.ar = init_vllm_registered_model(
        architectures=["MammothModa2ARForConditionalGeneration"],
        hf_config=cfg.llm_config,  # 使用 VL 子配置
    )
elif self.model_stage == "dit":
    self.dit = init_vllm_registered_model(
        architectures=["MammothModa2DiTPipeline"],
        hf_config=cfg,  # 使用顶层配置
    )
```

---

### 2.2 MammothModa2ARForConditionalGeneration

**位置**: `mammoth_moda2.py:544-687`

**继承**: `Qwen2_5_VLForConditionalGeneration`

**关键方法**:

#### forward()
```python
def forward(
    self,
    input_ids: torch.Tensor,      # [batch, seq_len]
    positions: torch.Tensor,       # [batch, seq_len]
    intermediate_tensors,          # PP 中间张量
    inputs_embeds: torch.Tensor,   # [batch, seq_len, hidden]
    **kwargs,
) -> OmniOutput:
```

**返回**: `OmniOutput(text_hidden_states, multimodal_outputs, intermediate_tensors)`

#### compute_logits()
```python
def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor:
    # 1. 调用父类计算 logits
    logits = super().compute_logits(hidden_states)
    
    # 2. 应用 T2I token 约束
    logits = self._apply_t2i_token_constraints(logits)
    
    return logits  # [batch, total_vocab_size]
```

#### _apply_t2i_token_constraints()
T2I 任务中强制 AR grid token 约束:
- **行尾**: 只允许采样 `eol_token_id`
- **行内**: 只允许采样 `visual_start` 到 `visual_end` 范围内的 token

---

### 2.3 MammothModa2Qwen2ForCausalLM

**位置**: `mammoth_moda2.py:255-536`

**职责**: MoE 语言模型核心

**关键属性**:
```python
# 词汇表
self.base_vocab_size: int          # 基础词汇量 (~152064)
self.gen_vocab_size: int           # 生成词汇量 (~32800)
self.total_vocab_size: int         # 总词汇量 (~184864)
self.gen_vocab_start_index: int    # 生成 token 起始 ID

# 嵌入层
self.embed_tokens: VocabParallelEmbedding      # 基础 token 嵌入
self.gen_embed_tokens: VocabParallelEmbedding  # 生成 token 嵌入 (可选)

# 输出头
self.lm_head: ParallelLMHead       # 基础 logits 输出
self.gen_head: ParallelLMHead      # 生成 logits 输出 (可选)

# Decoder 层
self.layers: List[Mammoth2DecoderLayer]
```

#### get_input_embeddings() - 双词汇表嵌入查找

```python
def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
    # input_ids: [batch, seq_len]
    
    # 1. 判断哪些是生成 token
    gen_mask = input_ids >= self.gen_vocab_start_index  # [batch, seq_len]
    
    if not gen_mask.any():
        # 全部是基础 token
        return self.embed_tokens(input_ids)
    
    if gen_mask.all():
        # 全部是生成 token
        gen_ids = input_ids - self.gen_vocab_start_index
        return self.gen_embed_tokens(gen_ids)
    
    # 2. 混合情况：分别处理
    out = torch.empty((num_tokens, hidden_size))
    out[~gen_mask] = self.embed_tokens(input_ids[~gen_mask])
    out[gen_mask] = self.gen_embed_tokens(gen_ids)
    
    return out  # [batch, seq_len, hidden_size]
```

#### forward() - 主前向传播

```python
def forward(
    self,
    input_ids: torch.Tensor,      # [batch, seq_len]
    positions: torch.Tensor,       # [batch, seq_len]
    intermediate_tensors,          # PP 中间张量
    inputs_embeds: torch.Tensor,   # [batch, seq_len, hidden_size]
) -> torch.Tensor | IntermediateTensors:
    
    # 1. 获取嵌入
    hidden_states = self.get_input_embeddings(input_ids)
    # hidden_states: [batch, seq_len, hidden_size]
    
    # 2. 生成 token mask (用于 MoE 路由)
    gen_token_mask = input_ids >= self.gen_vocab_start_index
    # gen_token_mask: [batch, seq_len] bool
    
    # 3. 逐层前向传播
    for layer in self.layers:
        hidden_states, residual = layer(
            positions, hidden_states, residual, gen_token_mask
        )
    
    # 4. 最终归一化
    hidden_states = self.norm(hidden_states, residual)
    
    return hidden_states  # [batch, seq_len, hidden_size]
```

#### compute_logits() - 拼接双词汇表 logits

```python
def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # 1. 基础 logits
    base_logits = self.logits_processor(self.lm_head, hidden_states)
    # base_logits: [batch, seq_len, base_vocab_size]
    
    # 2. 生成 logits
    gen_logits = self.gen_logits_processor(self.gen_head, hidden_states)
    # gen_logits: [batch, seq_len, gen_vocab_size]
    
    # 3. 拼接
    return torch.cat([base_logits, gen_logits], dim=-1)
    # output: [batch, seq_len, total_vocab_size]
```

---

### 2.4 Mammoth2DecoderLayer

**位置**: `mammoth_moda2.py:201-252`

**继承**: `Qwen2DecoderLayer`

**额外属性**:
```python
self.moe_enable: bool      # 当前层是否启用 MoE
self.gen_mlp: Qwen2MLP     # Generation Expert (可选)
```

#### forward() - MoE 路由前向传播

```python
def forward(
    self,
    positions: torch.Tensor,       # [batch, seq_len]
    hidden_states: torch.Tensor,   # [batch, seq_len, hidden_size]
    residual: torch.Tensor,        # [batch, seq_len, hidden_size]
    gen_token_mask: torch.Tensor,  # [batch, seq_len] bool
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # 1. Self-Attention
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    
    # 2. MoE Forward (FFN)
    hidden_states = moe_forward(
        hidden_states,
        self.mlp,           # Understanding Expert
        self.gen_mlp,       # Generation Expert
        gen_token_mask,
    )
    
    return hidden_states, residual
```

---

### 2.5 moe_forward() - MoE 路由函数

**位置**: `mammoth_moda2.py:75-156`

**功能**: 根据 token 类型路由到不同的专家网络

```python
def moe_forward(
    hidden_states: torch.Tensor,          # [batch, seq_len, hidden_size]
    und_expert: Callable,                 # Understanding Expert (mlp)
    gen_expert: Callable | None,          # Generation Expert (gen_mlp)
    gen_token_mask: torch.Tensor | None,  # [batch, seq_len] bool
) -> torch.Tensor:
    
    # Case 1: 无生成专家 → 全部走 understanding 专家
    if gen_expert is None:
        return und_expert(hidden_states)
    
    # Case 2: 无生成 token → 全部走 understanding 专家
    if gen_token_mask is None or not gen_token_mask.any():
        return und_expert(hidden_states)
    
    # Case 3: 全部是生成 token → 全部走 generation 专家
    if gen_token_mask.all():
        return gen_expert(hidden_states)
    
    # Case 4: 混合路由
    # 4.1 分离 token
    gen_pos = torch.where(flat_mask)[0]
    und_pos = torch.where(~flat_mask)[0]
    
    gen_hid = hidden_states[gen_pos]   # [N_gen, hidden_size]
    und_hid = hidden_states[und_pos]   # [N_und, hidden_size]
    
    # 4.2 分别通过专家
    gen_out = gen_expert(gen_hid)
    und_out = und_expert(und_hid)
    
    # 4.3 合并并恢复原始顺序
    merged = torch.cat([gen_out, und_out], dim=0)
    merged = merged[inverse_order]
    
    return merged  # [batch, seq_len, hidden_size]
```

---

## 3. 张量形状追踪

### 3.1 完整前向传播流程

```
输入 token_ids:                    [batch, seq_len]
                    │
                    ▼ embed_tokens / gen_embed_tokens
token_embeddings:                  [batch, seq_len, hidden_size]
                    │
                    ▼ Vision Tower (如果有图像输入)
vision_embeddings:                 [batch, num_patches, hidden_size]
                    │
                    ▼ 拼接文本和视觉嵌入
combined_embeddings:               [batch, total_seq_len, hidden_size]
                    │
                    ▼ N × Mammoth2DecoderLayer
                    │   每层:
                    │   - input_layernorm: [batch, seq_len, hidden_size]
                    │   - self_attn: [batch, seq_len, hidden_size]
                    │   - post_attention_layernorm: [batch, seq_len, hidden_size]
                    │   - moe_forward: [batch, seq_len, hidden_size]
                    ▼
hidden_states:                     [batch, seq_len, hidden_size]
                    │
                    ▼ RMSNorm
normalized_hidden:                 [batch, seq_len, hidden_size]
                    │
                    ▼ lm_head + gen_head
logits:                            [batch, seq_len, total_vocab_size]
```

### 3.2 MoE 路由细节

```
输入 hidden_states:                [batch, seq_len, hidden_size]
gen_token_mask:                    [batch, seq_len] (bool)

                    │ 展平
flat_hidden:                       [batch * seq_len, hidden_size]
flat_mask:                         [batch * seq_len] (bool)

                    │ 分离
gen_hidden:                        [N_gen_tokens, hidden_size]
und_hidden:                        [N_und_tokens, hidden_size]

                    │ 专家处理
gen_output:                        [N_gen_tokens, hidden_size]
und_output:                        [N_und_tokens, hidden_size]

                    │ 合并恢复顺序
merged_output:                     [batch * seq_len, hidden_size]
                    │ reshape
output:                            [batch, seq_len, hidden_size]
```

## 4. 配置参数

### 4.1 Mammothmoda2Qwen2_5_VLTextConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocab_size` | 152064 | 基础词汇表大小 |
| `hidden_size` | 8192 | 隐藏层维度 |
| `intermediate_size` | 29568 | FFN 中间层维度 |
| `num_hidden_layers` | 80 | Decoder 层数 |
| `num_attention_heads` | 64 | 注意力头数 |
| `num_key_value_heads` | 8 | KV 头数 (GQA) |
| `extra_gen_vocab` | True | 是否启用额外生成词汇表 |
| `gen_vocab_size` | 32800 | 生成词汇表大小 |
| `moe_type` | "ffn" | MoE 类型 |

### 4.2 moe_type 配置格式

| 格式 | 含义 |
|------|------|
| `"none"` | 不启用 MoE |
| `"attention"` | 仅注意力层启用 MoE |
| `"ffn"` | 仅 FFN 层启用 MoE |
| `"ffn_attention"` | 注意力和 FFN 都启用 MoE |
| `"ffn_attention-14:28"` | 第 14-27 层启用 MoE |

## 5. 权重映射

AR 阶段权重前缀映射 (checkpoint → vLLM):

| Checkpoint 前缀 | vLLM 前缀 |
|----------------|-----------|
| `llm_model.model.language_model.` | `language_model.` |
| `llm_model.model.visual.` | `visual.` |
| `llm_model.lm_head.` | `language_model.lm_head.` |
| `llm_model.gen_head.` | `language_model.gen_head.` |
| `llm_model.model.language_model.gen_embed_tokens.` | `language_model.gen_embed_tokens.` |
| `gen_transformer.` | 跳过 (DiT 阶段) |
| `gen_vae.` | 跳过 (DiT 阶段) |
