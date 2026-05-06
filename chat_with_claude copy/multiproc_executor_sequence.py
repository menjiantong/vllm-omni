"""
MultiprocDiffusionExecutor 通信机制可视化图
运行此脚本生成 PNG 图片 (需要 matplotlib)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_architecture():
    """绘制整体架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    # 标题
    ax.text(8, 11.5, 'MultiprocDiffusionExecutor 通信架构',
            fontsize=16, ha='center', fontweight='bold')

    # 主进程框
    main_process = FancyBboxPatch((5.5, 9), 5, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightblue',
                                   edgecolor='darkblue', linewidth=2)
    ax.add_patch(main_process)
    ax.text(8, 9.75, '主进程 (Executor)', fontsize=12, ha='center', fontweight='bold')

    # MessageQueue 组件
    # broadcast_mq
    broadcast_mq = FancyBboxPatch((1, 6.5), 3, 1.5,
                                   boxstyle="round,pad=0.05",
                                   facecolor='lightgreen',
                                   edgecolor='darkgreen', linewidth=2)
    ax.add_patch(broadcast_mq)
    ax.text(2.5, 7.25, 'broadcast_mq\n(Writer)', fontsize=10, ha='center')

    # result_mq
    result_mq = FancyBboxPatch((12, 6.5), 3, 1.5,
                                boxstyle="round,pad=0.05",
                                facecolor='lightyellow',
                                edgecolor='darkgoldenrod', linewidth=2)
    ax.add_patch(result_mq)
    ax.text(13.5, 7.25, 'result_mq\n(Reader)', fontsize=10, ha='center')

    # Worker 进程
    workers = []
    worker_positions = [(1, 2), (6, 2), (11, 2)]
    for i, (x, y) in enumerate(worker_positions):
        worker = FancyBboxPatch((x, y), 4, 3,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightcoral' if i == 0 else 'lightgray',
                                 edgecolor='darkred' if i == 0 else 'gray', linewidth=2)
        ax.add_patch(worker)
        ax.text(x + 2, y + 2.5, f'Worker {i}', fontsize=11, ha='center', fontweight='bold')
        ax.text(x + 2, y + 1.5, 'mq (Reader)', fontsize=9, ha='center')
        if i == 0:
            ax.text(x + 2, y + 0.7, 'result_mq (Writer)', fontsize=9, ha='center', color='darkgreen')
        workers.append(worker)

    # 绘制箭头
    # 主进程 -> broadcast_mq
    ax.annotate('', xy=(2.5, 8), xytext=(6.5, 9),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(4, 8.8, 'enqueue()', fontsize=9, color='green')

    # broadcast_mq -> Workers
    for i, (x, y) in enumerate(worker_positions):
        ax.annotate('', xy=(x + 2, 5), xytext=(2.5, 6.5),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Worker 0 -> result_mq
    ax.annotate('', xy=(12, 7.25), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2,
                               connectionstyle='arc3,rad=0.2'))
    ax.text(8.5, 5.5, 'enqueue(result)', fontsize=9, color='orange')

    # result_mq -> 主进程
    ax.annotate('', xy=(9.5, 9), xytext=(13.5, 8),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(12, 8.8, 'dequeue()', fontsize=9, color='orange')

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', label='broadcast_mq: 广播请求'),
        mpatches.Patch(facecolor='lightyellow', edgecolor='darkgoldenrod', label='result_mq: 返回结果'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='darkred', label='Worker 0: 返回结果'),
        mpatches.Patch(facecolor='lightgray', edgecolor='gray', label='Worker N: 仅执行'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)

    plt.tight_layout()
    plt.savefig('/home/mjt/project/vllm-omni/chat_with_claude/multiproc_architecture.png',
                dpi=150, bbox_inches='tight')
    print("架构图已保存到: multiproc_architecture.png")


def draw_sequence_diagram():
    """绘制时序图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(7, 9.5, '初始化与请求执行时序图', fontsize=14, ha='center', fontweight='bold')

    # 参与者
    participants = ['主进程', 'broadcast_mq', 'Worker 0', 'Worker N', 'result_mq']
    positions = [1.5, 4, 7, 10, 12.5]

    for name, x in zip(participants, positions):
        ax.plot([x, x], [1, 9], 'k-', lw=1, alpha=0.3)
        ax.text(x, 9, name, fontsize=10, ha='center', va='bottom', fontweight='bold')

    y = 8.5

    # 初始化阶段
    ax.text(7, y, '=== 初始化阶段 ===', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    y -= 0.6

    # 创建 MessageQueue
    ax.annotate('', xy=(4, y-0.3), xytext=(1.5, y-0.3),
                arrowprops=dict(arrowstyle='->', color='blue'))
    ax.text(2.75, y, 'create MessageQueue', fontsize=9, ha='center')
    y -= 0.6

    # 启动进程
    ax.annotate('', xy=(7, y-0.3), xytext=(1.5, y-0.3),
                arrowprops=dict(arrowstyle='->', color='blue'))
    ax.text(4.25, y, 'mp.Process.start()', fontsize=9, ha='center')
    y -= 0.6

    ax.annotate('', xy=(10, y-0.3), xytext=(1.5, y-0.3),
                arrowprops=dict(arrowstyle='->', color='blue'))
    ax.text(5.75, y, 'mp.Process.start()', fontsize=9, ha='center')
    y -= 0.6

    # Worker 连接
    ax.text(7, y, 'connect to broadcast_mq', fontsize=8, ha='center',
            color='blue', style='italic')
    ax.text(10, y, 'connect to broadcast_mq', fontsize=8, ha='center',
            color='blue', style='italic')
    y -= 0.6

    # 发送就绪信号 (通过 Pipe)
    ax.annotate('', xy=(1.5, y-0.3), xytext=(7, y-0.3),
                arrowprops=dict(arrowstyle='->', color='gray', ls='--'))
    ax.text(4.25, y, 'Pipe: send ready', fontsize=9, ha='center', color='gray')
    y -= 0.5

    ax.annotate('', xy=(1.5, y-0.3), xytext=(10, y-0.3),
                arrowprops=dict(arrowstyle='->', color='gray', ls='--'))
    ax.text(5.75, y, 'Pipe: send ready', fontsize=9, ha='center', color='gray')
    y -= 0.8

    # 运行阶段
    ax.text(7, y, '=== 运行阶段 ===', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    y -= 0.6

    # 发送请求
    ax.annotate('', xy=(4, y-0.3), xytext=(1.5, y-0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(2.75, y, 'enqueue(request)', fontsize=9, ha='center', color='green')
    y -= 0.5

    # 广播到 Worker
    ax.annotate('', xy=(7, y-0.3), xytext=(4, y-0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate('', xy=(10, y-0.3), xytext=(4, y-0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(5.5, y, 'dequeue()', fontsize=9, ha='center', color='green')
    y -= 0.5

    # 处理请求
    ax.text(7, y, 'process request', fontsize=8, ha='center',
            color='red', style='italic')
    ax.text(10, y, 'process request', fontsize=8, ha='center',
            color='red', style='italic')
    y -= 0.6

    # 只有 Worker 0 返回结果
    ax.annotate('', xy=(12.5, y-0.3), xytext=(7, y-0.3),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(9.75, y, 'enqueue(result)', fontsize=9, ha='center', color='orange')
    y -= 0.5

    # 主进程获取结果
    ax.annotate('', xy=(1.5, y-0.3), xytext=(12.5, y-0.3),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(7, y, 'dequeue(result)', fontsize=9, ha='center', color='orange')

    plt.tight_layout()
    plt.savefig('/home/mjt/project/vllm-omni/chat_with_claude/multiproc_sequence.png',
                dpi=150, bbox_inches='tight')
    print("时序图已保存到: multiproc_sequence.png")


def draw_ring_buffer():
    """绘制环形缓冲区示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 标题
    ax.text(7, 7.5, 'ShmRingBuffer 共享内存环形缓冲区', fontsize=14, ha='center', fontweight='bold')

    # 共享内存区域
    # 数据区
    ax.add_patch(FancyBboxPatch((1, 5), 12, 1.5, boxstyle="round,pad=0.02",
                                 facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax.text(7, 5.75, '数据区 (Data Section)', fontsize=11, ha='center', fontweight='bold')

    # Chunk 框
    chunk_positions = [1.5, 3.5, 5.5, 7.5, 9.5, 11]
    for i, x in enumerate(chunk_positions):
        ax.add_patch(FancyBboxPatch((x, 5.2), 1.5, 1, boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor='blue'))
        ax.text(x + 0.75, 5.7, f'chunk{i}', fontsize=8, ha='center')

    # 元数据区
    ax.add_patch(FancyBboxPatch((1, 3), 12, 1.5, boxstyle="round,pad=0.02",
                                 facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(7, 3.75, '元数据区 (Metadata Section)', fontsize=11, ha='center', fontweight='bold')

    # Metadata 框
    for i, x in enumerate(chunk_positions):
        ax.add_patch(FancyBboxPatch((x, 3.2), 1.5, 1, boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor='green'))
        ax.text(x + 0.75, 3.7, f'meta{i}', fontsize=8, ha='center')

    # 单个 metadata 结构
    ax.add_patch(FancyBboxPatch((1, 0.5), 12, 2, boxstyle="round,pad=0.02",
                                 facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax.text(7, 2.3, '单个 Metadata 结构 (1 + n_reader bytes)', fontsize=10, ha='center', fontweight='bold')

    # metadata 字段
    meta_fields = ['written\nflag', 'reader0\nflag', 'reader1\nflag', '...', 'readerN\nflag']
    meta_x = [1.5, 3.5, 5.5, 8, 10.5]
    for field, x in zip(meta_fields, meta_x):
        ax.add_patch(FancyBboxPatch((x, 0.7), 1.5, 1.2, boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor='orange'))
        ax.text(x + 0.75, 1.3, field, fontsize=7, ha='center')

    plt.tight_layout()
    plt.savefig('/home/mjt/project/vllm-omni/chat_with_claude/ring_buffer.png',
                dpi=150, bbox_inches='tight')
    print("环形缓冲区图已保存到: ring_buffer.png")


if __name__ == '__main__':
    try:
        draw_architecture()
        draw_sequence_diagram()
        draw_ring_buffer()
        print("\n所有图片已生成!")
    except Exception as e:
        print(f"生成图片失败: {e}")
        print("请确保安装了 matplotlib: pip install matplotlib")
