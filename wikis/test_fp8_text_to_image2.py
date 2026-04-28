#!/usr/bin/env python3
"""简单的 FP8 文生图测试脚本，带运维监控"""

import argparse
import time

import torch

from vllm_omni.entrypoints.omni import Omni


def get_memory_info() -> dict:
    """获取 GPU 显存信息"""
    if not torch.cuda.is_available():
        return {"available": False}

    allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
    peak = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

    return {
        "available": True,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "peak_gb": peak,
        "total_gb": total,
    }


def print_memory_stats(prefix: str = ""):
    """打印显存统计"""
    info = get_memory_info()
    if not info["available"]:
        return

    print(f"[显存] {prefix}: 已用 {info['allocated_gb']:.2f}GB / "
          f"峰值 {info['peak_gb']:.2f}GB / 总量 {info['total_gb']:.1f}GB")


def parse_args():
    parser = argparse.ArgumentParser(description="FP8 文生图测试")
    parser.add_argument("--prompt", default="a cup of coffee on the table", help="提示词")
    parser.add_argument("--output", default="coffee.png", help="输出文件名")
    parser.add_argument("--width", type=int, default=512, help="图片宽度")
    parser.add_argument("--height", type=int, default=512, help="图片高度")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-fp8", action="store_true", help="禁用 FP8 量化")
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B", help="模型名称")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("FP8 文生图测试")
    print("=" * 50)
    print(f"模型: {args.model}")
    print(f"提示词: {args.prompt}")
    print(f"尺寸: {args.width}x{args.height}")
    print(f"步数: {args.steps}")
    print(f"FP8: {'禁用' if args.no_fp8 else '启用'}")
    print("=" * 50)

    # 重置显存统计
    torch.cuda.reset_peak_memory_stats()

    # 加载模型
    print("\n[1/2] 加载模型...")
    t0 = time.perf_counter()

    quant_kwargs = {}
    if not args.no_fp8:
        quant_kwargs["quantization"] = "fp8"

    omni = Omni(
        model=args.model
        # ,
        # **quant_kwargs,
    )

    prompt = "a cup of coffee on the table"
    

    t1 = time.perf_counter()
    load_time = t1 - t0
    print(f"[耗时] 模型加载: {load_time:.2f}s")
    print_memory_stats("加载后")

    # 生成图片
    print("\n[2/2] 生成图片...")
    t0 = time.perf_counter()

    # outputs = omni.generate(
    #             prompts=list(args.prompt))
    outputs = omni.generate(prompt)
    images = outputs[0].request_output.images
    images[0].save("coffee.png")

    t1 = time.perf_counter()
    gen_time = t1 - t0
    print(f"[耗时] 图片生成: {gen_time:.2f}s")
    print_memory_stats("生成后")

    # 保存图片
    images = outputs[0].request_output.images
    images[0].save(args.output)
    print(f"\n[输出] 已保存到: {args.output}")

    # 汇总
    print("\n" + "=" * 50)
    print("运行汇总")
    print("=" * 50)
    print(f"模型加载耗时: {load_time:.2f}s")
    print(f"图片生成耗时: {gen_time:.2f}s")
    print(f"总耗时: {load_time + gen_time:.2f}s")
    print_memory_stats("最终")
    print("=" * 50)


if __name__ == "__main__":
    main()
