#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
FP8 文生图测试脚本

适用于 RTX 5060Ti 16GB 显卡
使用 FLUX.2-klein-4B 模型 + FP8 量化

模型信息：
- 模型: black-forest-labs/FLUX.2-klein-4B
- 参数量: 4B (最小)
- FP8 显存占用: ~8-10GB
- BF16 显存占用: ~16-18GB

使用方法:
    # 基础使用
    python test_fp8_text_to_image.py

    # 自定义提示词
    python test_fp8_text_to_image.py --prompt "a beautiful sunset over the ocean"

    # 使用 BF16 对比（需要更多显存）
    python test_fp8_text_to_image.py --no-fp8

    # 生成更大图片
    python test_fp8_text_to_image.py --width 1024 --height 1024
"""

import argparse
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FP8 文生图测试脚本 (FLUX.2-klein-4B)"
    )

    # 模型选择
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="模型名称。可选: FLUX.2-klein-4B (4B, 推荐), Tongyi-MAI/Z-Image-Turbo",
    )

    # 提示词
    parser.add_argument(
        "--prompt",
        default="A serene Japanese garden with cherry blossoms, a small wooden bridge over a koi pond, soft morning light, highly detailed, photorealistic",
        help="文本提示词",
    )

    # 输出设置
    parser.add_argument(
        "--output",
        default="fp8_output.png",
        help="输出图片路径",
    )

    # 图片尺寸
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="图片宽度 (默认 512，16GB 显卡推荐)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="图片高度 (默认 512，16GB 显卡推荐)",
    )

    # 推理步数
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="推理步数 (FLUX.2-klein 推荐 20-30)",
    )

    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于复现结果",
    )

    # FP8 量化开关
    parser.add_argument(
        "--no-fp8",
        action="store_true",
        help="禁用 FP8 量化，使用 BF16 (需要更多显存)",
    )

    # VAE 优化
    parser.add_argument(
        "--vae-slicing",
        action="store_true",
        help="启用 VAE slicing 节省显存",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="启用 VAE tiling 节省显存",
    )

    # 跳过层（用于调试）
    parser.add_argument(
        "--ignored-layers",
        type=str,
        default=None,
        help="跳过量化的层，逗号分隔。如: img_mlp,txt_mlp",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("FP8 文生图测试")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"提示词: {args.prompt[:60]}...")
    print(f"图片尺寸: {args.width}x{args.height}")
    print(f"推理步数: {args.steps}")
    print(f"FP8 量化: {'禁用' if args.no_fp8 else '启用'}")
    print("=" * 60)

    # 配置量化
    quant_kwargs = {}
    if not args.no_fp8:
        if args.ignored_layers:
            ignored = [s.strip() for s in args.ignored_layers.split(",") if s.strip()]
            quant_kwargs["quantization_config"] = {
                "method": "fp8",
                "ignored_layers": ignored,
            }
        else:
            quant_kwargs["quantization"] = "fp8"

    # 并行配置
    # parallel_config = DiffusionParallelConfig(
    #     ulysses_degree=1,
    #     ring_degree=1,
    #     cfg_parallel_size=1,
    #     tensor_parallel_size=1,
    #     vae_patch_parallel_size=1,
    # )

    # 初始化模型
    print("\n正在加载模型...")
    init_start = time.perf_counter()

    omni = Omni(
        model=args.model,
        # parallel_config=parallel_config,
        # vae_use_slicing=args.vae_slicing,
        # vae_use_tiling=args.vae_tiling,
        mode="text-to-image",
        **quant_kwargs,
    )

    init_end = time.perf_counter()
    print(f"模型加载完成，耗时: {init_end - init_start:.2f}s")

    # 创建随机数生成器
    generator = torch.Generator(
        device=current_omni_platform.device_type
    ).manual_seed(args.seed)

    # 记录显存使用
    torch.cuda.reset_peak_memory_stats()

    # 生成图片
    print("\n正在生成图片...")
    gen_start = time.perf_counter()

    outputs = omni.generate(
        {"prompt": args.prompt},
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=3.5,  # FLUX 推荐值
            generator=generator,
        ),
    )

    gen_end = time.perf_counter()
    gen_time = gen_end - gen_start

    # 获取显存使用
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB

    print(f"\n生成完成!")
    print(f"生成耗时: {gen_time:.2f}s")
    print(f"峰值显存: {peak_memory:.2f} GB")

    # 保存图片
    if not outputs or len(outputs) == 0:
        raise ValueError("生成失败，无输出")

    first_output = outputs[0]
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("输出格式错误")

    req_out = first_output.request_output
    if hasattr(req_out, "__iter__"):
        # 可能是列表
        for item in req_out:
            if hasattr(item, "images") and item.images:
                images = item.images
                break
        else:
            raise ValueError("未找到图片")
    elif hasattr(req_out, "images"):
        images = req_out.images
    else:
        raise ValueError(f"未知的输出格式: {type(req_out)}")

    if not images:
        raise ValueError("图片列表为空")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)

    print(f"\n图片已保存到: {output_path.absolute()}")

    # 打印总结
    print("\n" + "=" * 60)
    print("生成总结")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"量化: {'BF16' if args.no_fp8 else 'FP8'}")
    print(f"图片尺寸: {args.width}x{args.height}")
    print(f"推理步数: {args.steps}")
    print(f"生成耗时: {gen_time:.2f}s")
    print(f"峰值显存: {peak_memory:.2f} GB")
    print(f"输出文件: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
