"""
调试脚本：在 Qwen3Model.from_pretrained 相关位置打印 torch_dtype 处理日志

使用方法：
1. 在导入 transformers 之前设置环境变量或在代码中 monkey-patch
2. 运行模型加载代码

原理：
- torch_dtype 参数的工作流程：
  1. from_pretrained 解析参数 (modeling_utils.py:4628-4672)
  2. _get_dtype 确定 dtype (modeling_utils.py:1219-1296)
  3. _set_default_dtype 设置默认 dtype 用于模型初始化 (modeling_utils.py:2320-2344)
  4. 模型初始化时使用设置的默认 dtype
  5. _infer_parameter_dtype 推断参数应该转换的类型 (modeling_utils.py:627-660)
  6. _load_state_dict_into_meta_model 进行实际的类型转换 (modeling_utils.py:748-750)
"""

import logging

import torch
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

# 保存原始函数
_original_get_dtype = PreTrainedModel._get_dtype.__func__
_original_infer_parameter_dtype = PreTrainedModel._infer_parameter_dtype.__func__
_original_load_state_dict_into_meta_model = PreTrainedModel._load_state_dict_into_meta_model.__func__
_original_set_default_dtype = PreTrainedModel._set_default_dtype.__func__


def patched_get_dtype(cls, dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only):
    """打印 dtype 确定过程的日志"""
    print(f"\n{'=' * 60}")
    print("[DTYPE_DEBUG] _get_dtype called")
    print(f"[DTYPE_DEBUG]   Input dtype argument: {dtype}")
    print(f"[DTYPE_DEBUG]   Input dtype type: {type(dtype)}")

    result = _original_get_dtype(cls, dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only)

    config_out, dtype_out, dtype_orig = result
    print(f"[DTYPE_DEBUG]   Output config.dtype: {config_out.dtype}")
    print(f"[DTYPE_DEBUG]   Output dtype: {dtype_out}")
    print(f"[DTYPE_DEBUG]   Original default dtype: {dtype_orig}")
    print(f"{'=' * 60}\n")

    return result


def patched_set_default_dtype(cls, dtype):
    """打印设置默认 dtype 的日志"""
    print("\n[DTYPE_DEBUG] _set_default_dtype called")
    print(f"[DTYPE_DEBUG]   Setting default dtype to: {dtype}")
    print(f"[DTYPE_DEBUG]   Current torch.get_default_dtype(): {torch.get_default_dtype()}")

    result = _original_set_default_dtype(cls, dtype)

    print(f"[DTYPE_DEBUG]   After set, torch.get_default_dtype(): {torch.get_default_dtype()}")
    print(f"[DTYPE_DEBUG]   Returned original dtype: {result}")

    return result


def patched_infer_parameter_dtype(model, param_name, empty_param, keep_in_fp32_regex=None, hf_quantizer=None):
    """打印参数类型推断的日志"""
    result = _original_infer_parameter_dtype(model, param_name, empty_param, keep_in_fp32_regex, hf_quantizer)

    to_contiguous, casting_dtype = result
    # 只打印浮点类型的参数，避免日志过多
    if empty_param.dtype.is_floating_point:
        print(f"[DTYPE_DEBUG] _infer_parameter_dtype: {param_name[:50]}...")
        print(f"[DTYPE_DEBUG]   Weight file dtype (empty_param): {empty_param.dtype}")
        print(f"[DTYPE_DEBUG]   Model param dtype (old_param): {model.get_parameter_or_buffer(param_name).dtype}")
        print(f"[DTYPE_DEBUG]   Casting to: {casting_dtype}")

    return result


def patched_load_state_dict_into_meta_model(
    model,
    state_dict,
    shard_file,
    reverse_renaming_mapping,
    device_map=None,
    disk_offload_folder=None,
    disk_offload_index=None,
    hf_quantizer=None,
    keep_in_fp32_regex=None,
    device_mesh=None,
):
    """打印权重加载和类型转换的日志"""
    print("\n[DTYPE_DEBUG] _load_state_dict_into_meta_model called")
    print(f"[DTYPE_DEBUG]   Shard file: {shard_file}")
    print(f"[DTYPE_DEBUG]   Number of parameters to load: {len(state_dict)}")
    print(f"[DTYPE_DEBUG]   Model config dtype: {model.config.dtype}")

    result = _original_load_state_dict_into_meta_model(
        model,
        state_dict,
        shard_file,
        reverse_renaming_mapping,
        device_map,
        disk_offload_folder,
        disk_offload_index,
        hf_quantizer,
        keep_in_fp32_regex,
        device_mesh,
    )

    # 打印模型参数的最终 dtype 分布
    dtype_counts = {}
    for name, param in model.named_parameters():
        dtype_str = str(param.dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

    print(f"[DTYPE_DEBUG]   Final model dtype distribution: {dtype_counts}")

    return result


def apply_patches():
    """应用所有补丁"""
    PreTrainedModel._get_dtype = classmethod(patched_get_dtype)
    PreTrainedModel._set_default_dtype = classmethod(patched_set_default_dtype)
    PreTrainedModel._infer_parameter_dtype = patched_infer_parameter_dtype
    PreTrainedModel._load_state_dict_into_meta_model = patched_load_state_dict_into_meta_model
    print("[DTYPE_DEBUG] Patches applied successfully!")


def remove_patches():
    """移除所有补丁"""
    PreTrainedModel._get_dtype = classmethod(_original_get_dtype)
    PreTrainedModel._set_default_dtype = classmethod(_original_set_default_dtype)
    PreTrainedModel._infer_parameter_dtype = _original_infer_parameter_dtype
    PreTrainedModel._load_state_dict_into_meta_model = _original_load_state_dict_into_meta_model
    print("[DTYPE_DEBUG] Patches removed successfully!")


if __name__ == "__main__":
    # 示例用法
    print("""
    在你的代码中这样使用：

    ```python
    from debug_torch_dtype import apply_patches

    # 在加载模型之前应用补丁
    apply_patches()

    # 然后正常加载模型
    from transformers import Qwen3Model
    model = Qwen3Model.from_pretrained("path/to/model", torch_dtype=torch.float16)
    ```
    """)
