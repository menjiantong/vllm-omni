# Recipe 编写指南

本文档解释 `recipes/` 的定位，以及它与 `docs/` 和 `examples/` 的区别，并以 Qwen3-Omni 为例说明如何编写 recipe。

## 三种文档的定位

| 目录 | 定位 | 目标读者 | 内容特点 |
|------|------|----------|----------|
| `docs/` | **权威文档** | 产品用户 | 完整、详细、版本化，覆盖所有功能和参数 |
| `examples/` | **可运行代码** | 开发者 | 实际脚本、配置文件，可直接执行 |
| `recipes/` | **实战指南** | 运维/部署者 | 面向具体场景的"known-good"配置，快速上手 |

### `docs/` 的特点

- **权威性**: 产品官方文档，是功能的唯一真实来源
- **完整性**: 覆盖所有 API、参数、配置选项
- **版本化**: 与代码版本同步更新
- **示例**: `docs/user_guide/examples/online_serving/qwen3_omni.md`

以 Qwen3-Omni 为例，`docs/user_guide/examples/online_serving/qwen3_omni.md` 包含：
- 完整的安装说明
- 所有启动命令选项
- API 参考（模态控制、参数说明）
- OpenAI SDK 使用示例
- Gradio Demo 详细说明
- 嵌入的代码示例

### `examples/` 的特点

- **可执行**: 包含实际可运行的 Python 脚本和 Shell 脚本
- **代码导向**: 以代码为主，文档为辅
- **测试验证**: 通常用于 CI/CD 测试验证
- **示例**: `examples/online_serving/qwen3_omni/`

以 Qwen3-Omni 为例，`examples/online_serving/qwen3_omni/` 包含：
- `openai_chat_completion_client_for_multimodal_generation.py` - 实际的客户端代码
- `openai_realtime_client.py` - WebSocket 客户端
- `gradio_demo.py` - Web UI 演示
- `run_gradio_demo.sh` - 启动脚本
- `README.md` - 简要说明

### `recipes/` 的特点

- **场景导向**: 解决"如何在硬件 X 上运行模型 Y 完成任务 Z"
- **精简实用**: 只包含必要信息，快速上手
- **硬件绑定**: 明确测试过的硬件配置
- **链接引用**: 指向 `docs/` 和 `examples/` 而非重复内容
- **社区维护**: 由贡献者提交实测配置

## 以 Qwen3-Omni 为例

### docs 文档内容

`docs/user_guide/examples/online_serving/qwen3_omni.md`（约 250 行）包含：
- 安装说明
- 详细的服务启动命令（包括异步分块、自定义配置）
- 多模态请求示例（Python、curl）
- 模态控制详解（text、audio 输出控制）
- 流式输出说明
- Gradio Demo 完整文档
- 嵌入的代码示例

### examples 文档内容

`examples/online_serving/qwen3_omni/README.md`（约 460 行）包含：
- 部署参数调优（内存、并行度等）
- 多节点部署配置
- `/v1/realtime` WebSocket 客户端说明
- FAQ（模态控制、说话人选择）
- 更详细的命令行参数说明

### recipe 文档内容

`recipes/Qwen/Qwen3-Omni.md`（约 90 行）包含：
- 摘要表格（Vendor、Model、Task、Mode）
- 使用场景说明
- 参考链接（指向 docs 和 examples）
- 单一硬件配置（1x A100 80GB）
- 精简的启动命令
- 验证命令
- 简要注释

**关键区别**: recipe 不重复 docs 已有的内容，而是提供"已知可行"的配置快照，并链接到详细文档。

## 如何编写 Recipe

### 1. 确定文件位置

按厂商组织目录结构：
```
recipes/
  Qwen/
    Qwen3-Omni.md
    Qwen3-TTS.md
  Tencent-Hunyuan/
    HunyuanVideo.md
```

### 2. 使用模板

从 `recipes/TEMPLATE.md` 开始，填写以下部分：

#### Summary 表格

```markdown
- Vendor: 厂商名称
- Model: HuggingFace 模型 ID
- Task: 任务类型
- Mode: 部署模式（online serving / offline inference）
- Maintainer: 维护者（或 Community）
```

#### When to use this recipe

一句话描述适用场景。

#### References

链接到：
- `docs/` 下的权威文档
- `examples/` 下的可运行示例
- 相关 issue 或讨论

#### Hardware Support

按平台分节（GPU、ROCm、NPU），每个硬件配置包含：

- `#### Environment`: OS、Python、驱动、vLLM 版本
- `#### Command`: 启动命令
- `#### Verification`: 验证命令
- `#### Notes`: 内存占用、关键参数、已知限制

### 3. 编写原则

1. **精简**: 每个 hardware section 控制在 30-50 行
2. **链接导向**: 详细内容链接到 docs，代码示例链接到 examples
3. **实测验证**: 只写你亲自测试过的配置
4. **硬件明确**: 明确 GPU 型号、显存、驱动版本
5. **可复现**: 命令可直接复制执行

### 4. 示例对比

| 内容 | docs 包含 | examples 包含 | recipe 包含 |
|------|-----------|---------------|-------------|
| 完整 API 参考 | 是 | 否 | 否（链接到 docs） |
| 所有启动选项 | 是 | 部分 | 否（只写关键选项） |
| 可执行代码 | 否（嵌入展示） | 是 | 否（链接到 examples） |
| 硬件配置细节 | 否 | 否 | 是 |
| 验证命令 | 部分 | 部分 | 是（精简版） |
| 已知限制 | 部分 | 部分 | 是 |

## 提交 Recipe 的检查清单

- [ ] 文件位于正确的厂商目录下
- [ ] Summary 表格完整
- [ ] References 链接有效
- [ ] Environment 信息准确
- [ ] Command 可直接执行
- [ ] Verification 命令能验证服务正常
- [ ] Notes 注释实用
- [ ] 文档长度适中（不超过 100 行）
- [ ] 不重复 docs 已有的内容
