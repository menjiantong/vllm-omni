# Ovis-Image 权重加载问题分析与解决方案

## 一、问题现象

运行 Ovis-Image 模型时出现权重加载错误：

```
KeyError: 'transformer_blocks.0.ff.net.0.proj.bias'
```

完整错误堆栈：
```
File "vllm_omni/diffusion/models/ovis_image/ovis_image_transformer.py", line 677, in load_weights
    param = params_dict[name]
            ~~~~~~~~~~~^^^^^^
KeyError: 'transformer_blocks.0.ff.net.0.proj.bias'
```

---

## 二、问题原因

### 2.1 根本原因：权重命名不匹配

错误的核心在于 **safetensors 文件中的权重名** 与 **vLLM 模型中的参数名** 不一致。

| 来源 | 权重名示例 |
|------|-----------|
| safetensors 文件（diffusers 保存） | `transformer_blocks.0.ff.net.0.proj.weight` |
| vLLM 模型参数 | `transformer_blocks.0.ff.linear_in.weight` |

### 2.2 为什么两边命名不同？

**diffusers 的 FeedForward 结构：**
```python
# diffusers 源码结构
class FeedForward:
    def __init__(self):
        self.net = nn.ModuleList([
            GEGGLU(),      # net[0]，里面有 proj 层
            nn.Dropout(),  # net[1]，无参数
            nn.Linear(),   # net[2]，输出层
        ])
```
保存后的 key：
- `net.0.proj.weight/bias` - 输入投影层
- `net.2.weight/bias` - 输出层

**vLLM 的 FeedForward 结构：**
```python
# vllm_omni/diffusion/models/ovis_image/ovis_image_transformer.py
class OvisImageFeedForward(nn.Module):
    def __init__(self, ...):
        self.linear_in = MergedColumnParallelLinear(...)  # 对应 diffusers 的 net.0.proj
        self.act_fn = OvisImageSwiGLU()
        self.linear_out = RowParallelLinear(...)          # 对应 diffusers 的 net.2
```

参数名：
- `linear_in.weight/bias` - 输入投影层
- `linear_out.weight/bias` - 输出层

---

## 三、解决方案

在 `load_weights()` 方法中添加名字映射：

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    # ... 原有代码 ...

    # Weight name mapping from diffusers to vLLM naming convention
    weight_name_mapping = [
        (".ff.net.0.proj.", ".ff.linear_in."),
        (".ff.net.2.", ".ff.linear_out."),
        (".ff_context.net.0.proj.", ".ff_context.linear_in."),
        (".ff_context.net.2.", ".ff_context.linear_out."),
    ]

    params_dict = dict(self.named_parameters())
    
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        # Apply weight name mapping from diffusers to vLLM naming
        for old_name, new_name in weight_name_mapping:
            if old_name in name:
                name = name.replace(old_name, new_name)
                break

        # ... 后续加载逻辑 ...
```

### 映射关系表

| diffusers 权重名 | vLLM 权重名 | 说明 |
|---|---|---|
| `.ff.net.0.proj.` | `.ff.linear_in.` | FeedForward 输入投影层 |
| `.ff.net.2.` | `.ff.linear_out.` | FeedForward 输出层 |
| `.ff_context.net.0.proj.` | `.ff_context.linear_in.` | Context FeedForward 输入层 |
| `.ff_context.net.2.` | `.ff_context.linear_out.` | Context FeedForward 输出层 |

---

## 四、完整的权重加载流程

### 4.1 整体流程图

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 读取 safetensors 文件                               │
│  safetensors_weights_iterator()                             │
│  yield ("transformer_blocks.0.ff.net.0.proj.weight", tensor)│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 添加全局前缀 (ComponentSource.prefix)               │
│  "transformer." + name                                      │
│  → "transformer.transformer_blocks.0.ff.net.0.proj.weight"  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: load_weights() 中的名字映射                         │
│  ".ff.net.0.proj." → ".ff.linear_in."                       │
│  → "transformer.transformer_blocks.0.ff.linear_in.weight"   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: 在 params_dict 中查找并加载                         │
│  params_dict = dict(self.named_parameters())                │
│  param = params_dict[name]  # 找到模型参数                   │
│  weight_loader(param, loaded_weight)  # 复制数据            │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 各阶段详解

#### Step 1: Safetensors 文件中的 key 来源

**来源：原始训练框架保存时决定的**

当你使用 HuggingFace diffusers 保存模型时：
```python
model.save_pretrained("path")
```

保存时会调用 `model.state_dict()`，其中的 key 就是模型的参数名。所以 safetensors 里的 key **完全取决于原始代码如何定义模块结构**。

#### Step 2: 全局前缀添加

在 `DiffusersPipelineLoader` 中定义了 `ComponentSource`：

```python
# vllm_omni/diffusion/model_loader/diffusers_loader.py
@dataclasses.dataclass
class ComponentSource:
    model_or_path: str
    subfolder: str | None
    revision: str | None
    prefix: str = ""  # A prefix to prepend to all weights
```

对于 Ovis-Image，在 `pipeline_ovis_image.py` 中：
```python
self.weights_sources = [
    DiffusersPipelineLoader.ComponentSource(
        model_or_path=od_config.model,
        subfolder="transformer",
        prefix="transformer.",  # 添加这个前缀
    )
]
```

#### Step 3: 名字映射

在 `OvisImageTransformer2DModel.load_weights()` 中进行名字转换。

#### Step 4: 参数查找与加载

```python
params_dict = dict(self.named_parameters())  # 获取模型所有参数
param = params_dict[name]                     # 按 name 查找
weight_loader(param, loaded_weight)           # 复制权重数据
```

---

### 4.3 Prefix 参数的作用

**重要：prefix 主要是文档/调试用途，不直接决定参数名！**

```python
# vLLM 中定义层时
self.linear_in = MergedColumnParallelLinear(
    dim,
    [inner_dim, inner_dim],
    prefix=f"{prefix}.linear_in",  # 这个 prefix 只是记录"期望"的名字
)
```

**真正的参数名由模块层级决定：**

```
OvisImageTransformer2DModel
  └── transformer_blocks (ModuleList)
        └── [0] (OvisImageTransformerBlock)
              └── ff (OvisImageFeedForward)
                    └── linear_in (MergedColumnParallelLinear)
                          └── weight, bias (实际参数)
```

所以参数名是：`transformer_blocks.0.ff.linear_in.weight`

---

## 五、关键问题总结

| 问题 | 答案 |
|---|---|
| safetensors 的 key 从哪来？ | 原始训练代码保存时的 `state_dict()` 键名 |
| prefix 参数有什么用？ | 文档用途，不直接影响参数名 |
| 模型参数名怎么决定？ | 由模块层级结构决定（父模块名.子模块名.参数名） |
| 为什么需要映射？ | diffusers 和 vLLM 的模块结构/命名不同 |
| 如何加载？ | 名字映射 → params_dict 查找 → weight_loader 复制数据 |

---

## 六、相关代码文件

| 文件 | 作用 |
|---|---|
| `vllm_omni/diffusion/models/ovis_image/ovis_image_transformer.py` | Ovis-Image Transformer 模型定义，包含 `load_weights()` |
| `vllm_omni/diffusion/models/ovis_image/pipeline_ovis_image.py` | Ovis-Image Pipeline 定义，定义权重来源和前缀 |
| `vllm_omni/diffusion/model_loader/diffusers_loader.py` | 权重加载器，负责读取 safetensors 文件 |
| `vllm/model_executor/model_loader/weight_utils.py` | 权重加载工具函数，如 `safetensors_weights_iterator()` |
| `vllm/model_executor/layers/linear.py` | 线性层定义，包含 `QKVParallelLinear` 等 |
