# -*- coding: utf-8 -*-
"""
CSV 数据可视化脚本

功能：读取 step_service_data.csv，按 UAV 绘制每步处理 bits 折线图

用法：
  python visualize_csv.py --csv step_service_data.csv
  python visualize_csv.py --csv step_service_data_ep0.csv --output bits_chart.png
"""

import argparse
import os
import csv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="CSV 数据折线图")
    parser.add_argument("--csv", type=str, required=True,
                        help="step_service_data.csv 文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出图片路径（默认：csv同目录下同名.png）")
    return parser.parse_args()


def load_csv(csv_path):
    """读取 CSV，按 step 汇总每个 UAV 的总处理 bits"""
    # {uav_id: {step: total_bits}}
    data = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uav_id = int(row['uav_id'])
            step = int(row['step'])
            bits = float(row['processed_bits'])

            if uav_id not in data:
                data[uav_id] = {}
            data[uav_id][step] = data[uav_id].get(step, 0.0) + bits

    return data


def plot(data, output_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # 设置中文字体（Windows）
        matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("[错误] 请先安装 matplotlib: pip install matplotlib")
        return

    # 颜色与样式
    uav_colors = ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b']
    uav_ids = sorted(data.keys())

    # 确定全局 step 范围
    all_steps = sorted({s for uid in uav_ids for s in data[uid].keys()})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                    facecolor='#1a1a2e',
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('UAV 每步处理数据量', color='white', fontsize=14, fontweight='bold', y=0.98)

    # ---- 上图：原始折线 ----
    ax1.set_facecolor('#16213e')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('#444')
    ax1.spines['left'].set_color('#444')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for uid in uav_ids:
        color = uav_colors[uid % len(uav_colors)]
        steps = sorted(data[uid].keys())
        bits  = [data[uid][s] for s in steps]
        ax1.plot(steps, bits, color=color, lw=1.5, alpha=0.9,
                 label=f'UAV {uid}')
        # 服务步骤高亮（bits > 0 的点）
        active_steps = [s for s in steps if data[uid][s] > 0]
        active_bits  = [data[uid][s] for s in active_steps]
        ax1.scatter(active_steps, active_bits, color=color, s=8, alpha=0.5, zorder=3)

    ax1.set_xlabel('Step', color='white', fontsize=11)
    ax1.set_ylabel('处理量 (bits)', color='white', fontsize=11)
    ax1.set_title('每步处理 bits（原始）', color='#aaa', fontsize=10, pad=6)
    ax1.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    ax1.grid(True, alpha=0.15, color='gray')

    # ---- 下图：累计折线 ----
    ax2.set_facecolor('#16213e')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#444')
    ax2.spines['left'].set_color('#444')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for uid in uav_ids:
        color = uav_colors[uid % len(uav_colors)]
        steps = sorted(data[uid].keys())
        cumsum = np.cumsum([data[uid][s] for s in steps])
        ax2.plot(steps, cumsum, color=color, lw=2, linestyle='--',
                 label=f'UAV {uid} 累计')
        ax2.fill_between(steps, cumsum, alpha=0.1, color=color)

    ax2.set_xlabel('Step', color='white', fontsize=11)
    ax2.set_ylabel('累计处理量 (bits)', color='white', fontsize=11)
    ax2.set_title('累计处理 bits', color='#aaa', fontsize=10, pad=6)
    ax2.legend(fontsize=10, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    ax2.grid(True, alpha=0.15, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"[OK] 图表已保存: {output_path}")

    # 打印简单统计
    print("\n=== 统计摘要 ===")
    for uid in uav_ids:
        total = sum(data[uid].values())
        active = sum(1 for v in data[uid].values() if v > 0)
        # 正确的单位转换：bits -> MB (bits / (1024 * 1024 * 8))
        total_mb = total / (1024 * 1024 * 8)
        print(f"  UAV {uid}: 总处理 {total:.0f} bits ({total_mb:.2f} MB)"
              f"  |  有效服务步数: {active} / {len(data[uid])}")


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"找不到 CSV 文件: {args.csv}")

    # 默认输出路径：csv同目录下同名.png
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.csv)[0]
        output_path = base + '_bits_chart.png'

    data = load_csv(args.csv)
    print(f"[OK] 读取 CSV: {args.csv}")
    print(f"     包含 UAV: {sorted(data.keys())}")
    print(f"     Step 范围: {min(s for d in data.values() for s in d)} "
          f"~ {max(s for d in data.values() for s in d)}")

    plot(data, output_path)


if __name__ == "__main__":
    main()
