,
        img_token: str = "<|image token|>",
        boi_token: str = "<|image start|>",
        eoi_token: str = "<|image end|>",
        eol_token: str = "<|endofline|>",
        eof_token: str = "<|endoffile|>",
        **kwargs,
    ):
```

### 2.3 词汇表结构

```
总词汇表大小: ~184864

┌─────────────────────────────────────────────────────────────────────┐
│                        词汇表结构                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [0, 151642)         mergeable_ranks (BPE 基础词汇)                 │
│                      - 标准 tiktoken BPE token                      │
│                                                                     │
│  [151643, 151645)    核心特殊 token                                  │
│                      - ENDOFTEXT, IMSTART, IMEND                    │
│                                                                     │
│  [151645, 151664)    Qwen 特殊 token                                 │
│                      - <|object_ref_start|>, <|box_start|>, ...    │
│                      - <|vision_start|>, <|vision_end|>, ...       │
│                      - <|image_pad|>, <|video_pad|>                │
│                                                                     │
│  [151664, 152064)    扩展 token                                      │
│                      - <|extra_0|> ... <|extra_180|>               │
│                      - <|extra_margin_0|> ...                       │
│                      - <|endofline|>, <|endoffile|>, ...           │
│                                                                     │
│  [152064, 184864)    生成词汇表 (gen_vocab)                          │
│                      - 视觉 token (用于 AR 生成)                    │
│                      - gen_vocab_size = 32800                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 关键属性

```python
# 视觉相关 token
self.image_content_token = "<|image_pad|>"      # Qwen2.5-VL 兼容
self.gen_image_token = "<|gen_image_pad|>"
self.gen_image_placeholder_token = "<|gen_placeholder|>"

# 视觉 token ID 列表
self.visual_tokens = [
    "<|image_pad|>",
    "<|video_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
]
self.visual_tokens_ids = [151655, 151656, 151652, 151653]

# 视觉范围
self.vision_range = (boi_token_id, vocab_size - 1)
# 例: (151657, 184863)
```

### 2.5 特殊 Token ID 表

| Token | ID | 用途 |
|-------|-----|------|
| `ENDOFTEXT` | 151643 | 文本结束 |
| `IMSTART` | 151644 | 对话开始 |
| `IMEND` | 151645 | 对话结束 |
| `<|vision_start|>` | 151652 | 视觉输入开始 |
| `<|vision_end|>` | 151653 | 视觉输入结束 |
| `<|image_pad|>` | 151655 | 图像填充 token |
| `<|video_pad|>` | 151656 | 视频填充 token |
| `<|endofline|>` | 184862 | AR grid 行尾 |
| `<|endoffile|>` | 184863 | AR grid 结束 |

### 2.6 编码示例

```python
tokenizer = MammothUTokenizer(...)

# 文本编码
tokens = tokenizer.tokenize("画一只猫")
# → [b'\xe7\x94\xbb', b'\xe4\xb8\x80', b'\xe5\x8f\xaa', b'\xe7\x8c\xab']
# → IDs: [12345, 67890, 11111, 22222]

# 特殊 token 编码
tokens = tokenizer.tokenize("<|image_pad|>")
# → ["<|image_pad|>"]
# → ID: 151655

# 生成 token (通过 AR 模型生成)
gen_token_id = 152064 + 123  # 在 gen_vocab 范围内
# → 对应某个视觉特征
```

### 2.7 解码

```python
# 标准解码
text = tokenizer.decode([12345, 67890])
# → "画一"

# 跳过特殊 token
text = tokenizer.decode([151655, 12345], skip_special_tokens=True)
# → 只解码 12345，跳过 <|image_pad|>

# 生成 token 解码
# 通常生成 token 不直接解码为文本，而是用于 DiT 阶段的条件
```

### 2.8 与 Qwen2.5-VL 兼容性

MammothUTokenizer 保持与 Qwen2.5-VL tokenizer 的兼容性：

```python
# 相同的特殊 token
image_token_id = 151655  # <|image_pad|>
video_token_id = 151656  # <|video_pad|>
vision_start_token_id = 151652
vision_end_token_id = 151653

# 相同的分词算法
PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
```

---

## 3. 位置编码使用场景

### 3.1 AR 阶段

AR 阶段使用 **标准 Qwen2.5 RoPE**，不需要 3D 编码：

```python
# Qwen2.5 使用 1D 位置编码
positions = torch.arange(seq_len)
# 通过 Qwen2Attention 内部的 rotary_emb 处理
```

### 3.2 DiT 阶段

DiT 阶段使用 **3D 实值 RoPE**：

```python
# 预计算频率 (在 DiTPipeline.__init__ 中)
self.gen_freqs_cis = RotaryPosEmbedReal.get_freqs_real(
    axes_dim=(40, 40, 40),
    axes_lens=(10000, 10000, 10000),
    theta=10000,
)

# 推理时构建位置编码
(context_rotary, ref_rotary, noise_rotary, rotary_emb,
 encoder_seq_lens, seq_lens) = self.rope_embedder(
    self.gen_freqs_cis,
    text_attention_mask,
    l_effective_ref_img_len,
    l_effective_img_len,
    ref_img_sizes,
    img_sizes,
    device,
)

# 在 Attention 中应用
query = apply_real_rotary_emb(query, freqs_cos, freqs_sin)
key = apply_real_rotary_emb(key, freqs_cos, freqs_sin)
```

---

## 4. 配置参数关系

### 4.1 AR 阶段 RoPE 配置

```python
# 在 text_config 中
rope_theta: float = 1000000.0   # Qwen2.5 使用较大的 theta
rope_scaling: dict | None = None
```

### 4.2 DiT 阶段 RoPE 配置

```python
# 在顶层 Mammothmoda2Config 中
gen_axes_dim_rope: [40, 40, 40]    # 各轴维度
gen_axes_lens: [10000, 10000, 10000]  # 各轴最大长度

# 在 gen_dit_config 中
axes_dim_rope: (32, 32, 32)        # 必须满足 head_dim = sum(axes_dim_rope)
axes_lens: (300, 512, 512)         # 可能被顶层 gen_axes_lens 覆盖
```

### 4.3 维度一致性检查

```python
# Transformer2DModel.__init__ 中
if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
    raise ValueError(
        f"hidden_size // num_attention_heads ({hidden_size // num_heads}) "
        f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
    )
```
