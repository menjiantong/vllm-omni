# Transformer2DModel.load_weights 完整解析

## 1. `named_parameters()` 得到什么

```python
params_dict = dict(self.named_parameters())
```

`self.named_parameters()` 递归遍历 `Transformer2DModel` 的**所有子模块**，返回 `(name, param)` 对，其中 `name` 是用 `.` 连接的属性路径。例如：

```
layers.0.attn.to_q.weight              # 非TP模式
layers.0.attn.to_qkv.weight            # TP模式
layers.0.feed_forward.linear_1.weight  # 非TP模式
layers.0.feed_forward.w13.weight       # TP模式
layers.0.feed_forward.w2.weight        # TP模式
layers.0.norm1.linear.weight
x_embedder.weight
noise_refiner.0.attn.to_q.weight
...
```

关键点：**name 来自 Python 属性路径**，即 `self.layers[0].attn.to_q.weight` -> `layers.0.attn.to_q.weight`。

## 2. 为什么是 `.feed_forward.linear_1.` 而不是 `layers`？

这是两个不同层级的东西：

- **checkpoint 的 key** 包含完整路径：`layers.0.feed_forward.linear_1.weight`
- **`stacked_params_mapping`** 中的 `.feed_forward.linear_1.` 是**子串匹配模式**，只匹配 key 中的这一段

`TransformerBlock` 的 `__init__`（mammothmoda2_dit_model.py line 681）定义了：

```python
self.feed_forward = LuminaFeedForward(...)
```

而 `LuminaFeedForward` 在非 TP 模式下（line 206-208）定义了：

```python
self.linear_1 = nn.Linear(...)  # gate
self.linear_2 = nn.Linear(...)  # down
self.linear_3 = nn.Linear(...)  # up
```

所以 `self.layers[0].feed_forward.linear_1.weight` 是完全合法的参数路径。

`Transformer2DModel` 本身没有 `feed_forward`，但它的 `self.layers` 是 `nn.ModuleList[TransformerBlock]`，每个 `TransformerBlock` 有 `self.feed_forward`。`named_parameters()` 是**递归**的，会展开所有子模块。

## 3. `attn` 的 key 哪里来的

同理，`TransformerBlock.__init__`（line 654/664）定义了 `self.attn`：

```python
# TP模式
self.attn = TPAttention(...)       # 有 self.to_qkv (QKVParallelLinear)

# 非TP模式
self.attn = Attention(...)         # 有 self.to_q, self.to_k, self.to_v
```

- **非 TP 模式**：diffusers 的 `Attention` 有 `self.to_q`, `self.to_k`, `self.to_v` 属性
- **TP 模式**：`TPAttention` 有 `self.to_qkv`（`QKVParallelLinear`）

所以 checkpoint 中的 key `layers.0.attn.to_q.weight` 对应 `self.layers[0].attn.to_q.weight`。

## 4. `load_weights` 整体流程

### 4.1 Checkpoint 来源

HuggingFace safetensors 文件，原始 key 格式：

```
gen_transformer.layers.0.attn.to_q.weight        # 原始非TP权重
gen_transformer.layers.0.feed_forward.linear_1.weight
gen_transformer.layers.0.feed_forward.linear_3.weight
gen_vae.*
llm_model.*                                       # LLM权重，会被丢弃
```

### 4.2 加载链路

```
safetensors文件
  |
  v  safetensors_weights_iterator 读取 (key, tensor) 对
MammothModa2DiTPipeline.load_weights
  |
  v  hf_to_vllm_mapper 丢弃 llm_model.* 前缀
AutoWeightsLoader.load_weights
  |
  v  按 gen_transformer. 前缀分组，递归加载子模块
  v  去掉 gen_transformer. 前缀后传给 Transformer2DModel.load_weights
Transformer2DModel.load_weights
  |
  v  遍历每个 (name, weight)
  v  如果 tp_size > 1，做 stacked mapping：
  v    feed_forward.linear_1 + linear_3 -> feed_forward.w13  (合并gate+up)
  v    attn.to_q + to_k + to_v -> attn.to_qkv              (合并QKV)
  v  否则直接按 name 匹配 params_dict 加载
```

### 4.3 核心逻辑（mammothmoda2_dit_model.py line 1159-1187）

```python
for name, loaded_weight in weights_list:
    # 1. 跳过 llm_model.* 权重
    if name.startswith("llm_model."):
        continue

    # 2. TP模式：检查是否需要 stack
    for param_name, weight_name, shard_id in stacked_params_mapping:
        if weight_name in name:  # 子串匹配，如 ".feed_forward.linear_1." in name
            new_name = name.replace(weight_name, param_name)  # 替换为 TP 参数名
            # 例如: layers.0.feed_forward.linear_1.weight
            #    -> layers.0.feed_forward.w13.weight
            weight_loader(param, loaded_weight, shard_id)  # shard_id=0 表示第0个分片
            matched = True
            break

    # 3. 非TP模式或未匹配的权重：直接加载
    if name in params_dict:
        weight_loader(param, loaded_weight)
```

### 4.4 为什么要做 stacked mapping？

因为 HuggingFace checkpoint 保存的是**非 TP** 的权重结构（`linear_1`, `linear_3`, `to_q`, `to_k`, `to_v`），但 vLLM 在 TP 模式下构建的是**合并后的并行层**（`w13`, `to_qkv`）。`load_weights` 的职责就是把原始的分开权重正确地拼装/分片到 TP 并行参数中。

## 5. 权重映射对照表

### 5.1 FFN（LuminaFeedForward）

| Checkpoint Key（非TP） | TP模式参数名 | 合并说明 |
|---|---|---|
| `*.feed_forward.linear_1.weight` | `*.feed_forward.w13.weight` | gate投影，shard_id=0 |
| `*.feed_forward.linear_3.weight` | `*.feed_forward.w13.weight` | up投影，shard_id=1 |
| `*.feed_forward.linear_2.weight` | `*.feed_forward.w2.weight` | down投影，直接加载 |

SwiGLU 计算：`output = down(silu(gate(x)) * up(x))`

### 5.2 Attention

| Checkpoint Key（非TP） | TP模式参数名 | 合并说明 |
|---|---|---|
| `*.attn.to_q.weight` | `*.attn.to_qkv.weight` | Query，shard_id="q" |
| `*.attn.to_k.weight` | `*.attn.to_qkv.weight` | Key，shard_id="k" |
| `*.attn.to_v.weight` | `*.attn.to_qkv.weight` | Value，shard_id="v" |
| `*.attn.to_out.0.weight` | `*.attn.to_out.0.weight` | 输出投影，直接加载 |

### 5.3 其他权重（直接加载，无需合并）

| Checkpoint Key | 说明 |
|---|---|
| `x_embedder.weight` | 输入patch嵌入 |
| `ref_image_patch_embedder.weight` | 参考图patch嵌入 |
| `time_caption_embed.*.weight` | 时间步+文本嵌入 |
| `rope_embedder.*` | RoPE位置编码 |
| `norm_out.*.weight` | 输出归一化+投影 |
| `image_index_embedding` | 参考图索引嵌入 |
| `*.norm1.linear.weight` | TransformerBlock自适应归一化 |
| `*.norm2.weight` | TransformerBlock后注意力归一化 |
| `*.ffn_norm1.weight` | TransformerBlock前FFN归一化 |
| `*.ffn_norm2.weight` | TransformerBlock后FFN归一化 |
| `*.attn.norm_q.weight` | Query归一化 |
| `*.attn.norm_k.weight` | Key归一化 |

## 6. `_hsdp_shard_conditions` 的作用（HSDP 分片）

### 6.1 什么是 HSDP？

HSDP = **Hybrid Sharded Data Parallel**（混合分片数据并行），是一种分布式训练/推理技术：

- **Shard（分片）**：将模型参数切分到多个 GPU，每个 GPU 只保存一部分参数
- **Replicate（复制）**：多组 GPU 各持有一份完整参数副本

```
假设 8 GPU，shard_size=2，replicate_size=4：

GPU0 GPU1 | GPU2 GPU3 | GPU4 GPU5 | GPU6 GPU7
  分片1   |   分片2   |   分片3   |   分片4
  ----第1组复制----  ----第2组复制----
```

### 6.2 `_is_transformer_block` 的作用

```python
@staticmethod
def _is_transformer_block(name: str, module) -> bool:
    return (
        ("layers" in name or "noise_refiner" in name or "context_refiner" in name or "ref_image_refiner" in name)
        and name.split(".")[-1].isdigit()
    )

_hsdp_shard_conditions = [_is_transformer_block]
```

这个函数判断一个模块**是否应该被 HSDP 分片**：

- **条件**：`name` 包含 `layers`/`noise_refiner`/`context_refiner`/`ref_image_refiner`，且最后一段是数字
- **匹配示例**：
  - `layers.0` -> True（TransformerBlock，需要分片）
  - `layers.12` -> True
  - `noise_refiner.1` -> True
  - `x_embedder` -> False（嵌入层，不分片）
  - `time_caption_embed` -> False
  - `layers.0.attn` -> False（最后一段不是数字）

### 6.3 HSDP 分片流程（hsdp.py line 157-202）

```python
hsdp_shard_conditions = getattr(model, "_hsdp_shard_conditions", None)

# 遍历所有子模块，对满足条件的模块调用 fully_shard
for name, module in reversed(list(model.named_modules())):
    if any(cond(name, module) for cond in hsdp_shard_conditions):
        fully_shard(module, reshard_after_forward=..., mesh=..., mp_policy=...)
```

**效果**：每个 `TransformerBlock` 被独立分片，实现细粒度的 HSDP 并行。

### 6.4 为什么只分片 TransformerBlock？

| 模块类型 | 是否分片 | 原因 |
|---|---|---|
| `TransformerBlock` (layers.*) | 是 | 参数量大，分片收益高 |
| `x_embedder` | 否 | 参数量小，复制更高效 |
| `time_caption_embed` | 否 | 计算密集度低 |
| `norm_out` | 否 | 输出层，保持一致性 |

## 7. `packed_modules_mapping` 的作用（LoRA 权重映射）

### 7.1 定义

```python
packed_modules_mapping = {
    "feed_forward.w13": ["feed_forward.linear_1", "feed_forward.linear_3"],
    "attn.to_qkv": ["attn.to_q", "attn.to_k", "attn.to_v"],
}
```

### 7.2 使用场景：LoRA 加载

LoRA checkpoint 通常保存的是**原始（非TP）权重名**：
- `attn.to_q.weight`
- `attn.to_k.weight`
- `feed_forward.linear_1.weight`

但 TP 模型使用的是**合并后的参数名**：
- `attn.to_qkv.weight`
- `feed_forward.w13.weight`

`packed_modules_mapping` 用于**扩展 LoRA 支持的模块名**（lora/utils.py line 30-55）：

```python
def _expand_expected_modules_for_packed_layers(
    supported_modules: set[str],
    packed_modules_mapping: dict[str, list[str]] | None,
) -> set[str]:
    expanded = set(supported_modules)
    for packed_name, sub_names in packed_modules_mapping.items():
        if packed_name in supported_modules:
            expanded.update(sub_names)  # 添加子模块名
    return expanded

# 示例：
# 输入: supported_modules = {"attn.to_qkv", "feed_forward.w13"}
# 输出: {"attn.to_qkv", "attn.to_q", "attn.to_k", "attn.to_v",
#        "feed_forward.w13", "feed_forward.linear_1", "feed_forward.linear_3"}
```

### 7.3 与 `stacked_params_mapping` 的区别

| 属性 | 定义位置 | 用途 | 使用时机 |
|---|---|---|---|
| `packed_modules_mapping` | 类属性 | LoRA 权重名扩展 | 加载 LoRA checkpoint |
| `stacked_params_mapping` | `load_weights` 内部 | 主模型权重合并 | 加载主模型 checkpoint |

两者映射关系相同，但用于不同阶段：
- **主模型加载**：`load_weights` 用 `stacked_params_mapping` 把分开的权重合并到 TP 参数
- **LoRA 加载**：LoRA manager 用 `packed_modules_mapping` 识别原始权重名对应的 LoRA 适配器

## 8. 完整的模型子模块结构

```
Transformer2DModel
 |-- rope_embedder: RotaryPosEmbedReal
 |-- x_embedder: nn.Linear
 |-- ref_image_patch_embedder: nn.Linear
 |-- time_caption_embed: Lumina2CombinedTimestepCaptionEmbedding
 |     |-- time_proj: Timesteps
 |     |-- timestep_embedder: TimestepEmbedding
 |     |-- caption_embedder: Sequential(RMSNorm, Linear)
 |-- noise_refiner: ModuleList[TransformerBlock] x num_refiner_layers
 |-- ref_image_refiner: ModuleList[TransformerBlock] x num_refiner_layers
 |-- context_refiner: ModuleList[TransformerBlock] x num_refiner_layers
 |-- layers: ModuleList[TransformerBlock] x num_layers
 |     |-- [i].attn: TPAttention (TP) / Attention (非TP)
 |     |     |-- TP: to_qkv, norm_q, norm_k, to_out[RowParallelLinear], attn
 |     |     |-- 非TP: to_q, to_k, to_v, norm_q, norm_k, to_out[Linear]
 |     |-- [i].feed_forward: LuminaFeedForward
 |     |     |-- TP: w13(MergedColumnParallelLinear), act(SiluAndMul), w2(RowParallelLinear)
 |     |     |-- 非TP: linear_1(gate), linear_2(down), linear_3(up)
 |     |-- [i].norm1: LuminaRMSNormZero (modulation=True) / RMSNorm (modulation=False)
 |     |-- [i].norm2: RMSNorm
 |     |-- [i].ffn_norm1: RMSNorm
 |     |-- [i].ffn_norm2: RMSNorm
 |-- norm_out: LuminaLayerNormContinuous
 |-- image_index_embedding: Parameter(5, hidden_size)
```
