"""
RMAPPO 无人机处理数据可视化脚本
绘制风格与MADDPG和MATD3一致
读取 results/MyEnv/UAV_EdgeComputing/rmappo/check/runXX/episode_data_processed.csv 文件
数据已经转换为MB单位
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def find_latest_csv(base_dir):
    """找到最新的 episode_data_processed.csv 文件"""
    csv_files = list(Path(base_dir).glob('**/episode_data_processed.csv'))
    if not csv_files:
        return None
    # 按修改时间排序，返回最新的
    return max(csv_files, key=os.path.getmtime)


def load_processed_data(csv_file):
    """从 CSV 文件加载处理数据（已经是MB单位）"""
    try:
        df = pd.read_csv(csv_file)
        episodes = df['episode'].values

        # 提取UAV处理数据（已经是MB）
        uav_data = {}
        for col in df.columns:
            if col.startswith('uav_') and col.endswith('_processed_mb'):
                uav_name = col.replace('_processed_mb', '')
                uav_data[uav_name] = df[col].values

        # 提取总处理数据
        total_data = df['total_processed_mb'].values if 'total_processed_mb' in df.columns else None

        return episodes, uav_data, total_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def calculate_moving_average(data, window=50):
    """计算移动平均"""
    if len(data) < window:
        window = len(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_uav_processed_data(csv_file, output_dir=None):
    """绘制无人机处理数据曲线（与MADDPG/MATD3风格一致）"""
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)

    # 加载数据
    episodes, uav_data, total_data = load_processed_data(csv_file)

    if episodes is None or len(episodes) == 0:
        print(f"No data found in {csv_file}")
        return

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 颜色定义
    colors = ['#1f77b4', '#ff7f0e', '#d62728']  # UAV0, UAV1, Sum

    # 第一个子图：原始处理数据
    ax1 = axes[0]
    for idx, (uav_name, data) in enumerate(uav_data.items()):
        ax1.plot(episodes, data, label=uav_name, color=colors[idx], linewidth=1.5, alpha=0.7)

    # 绘制总和线
    if total_data is not None:
        ax1.plot(episodes, total_data, label='Sum', color=colors[2], linewidth=2, linestyle='--', alpha=0.8)

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Processed Data (MB)', fontsize=12)
    ax1.set_title('UAV Processed Data (Raw)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # 第二个子图：平滑后的处理数据（移动平均）
    ax2 = axes[1]

    # 计算移动平均
    window_50 = 50
    for idx, (uav_name, data) in enumerate(uav_data.items()):
        ma_data = calculate_moving_average(data, window=window_50)
        episodes_ma = episodes[window_50-1:] if len(episodes) > window_50-1 else episodes
        ax2.plot(episodes_ma, ma_data, label=f'{uav_name} (MA-50)', color=colors[idx], linewidth=2)

    # 总和的移动平均
    if total_data is not None:
        sum_ma = calculate_moving_average(total_data, window=window_50)
        episodes_ma = episodes[window_50-1:] if len(episodes) > window_50-1 else episodes
        ax2.plot(episodes_ma, sum_ma, label='Sum (MA-50)', color=colors[2], linewidth=2.5, linestyle='--', alpha=0.8)

    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Processed Data (MB)', fontsize=12)
    ax2.set_title('UAV Processed Data (Smoothed with MA-50)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = os.path.join(output_dir, f'uav_processed_data_visualization_{timestamp}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ UAV processed data plot saved to: {output_file}")

    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"UAV Processed Data Statistics (in MB):")
    print(f"{'='*60}")
    print(f"Total Episodes: {len(episodes)}")
    for uav_name, data in uav_data.items():
        print(f"\n{uav_name}:")
        print(f"  Min: {np.min(data):.2f} MB")
        print(f"  Max: {np.max(data):.2f} MB")
        print(f"  Mean: {np.mean(data):.2f} MB")
        print(f"  Std Dev: {np.std(data):.2f} MB")
        print(f"  Last 50 Episodes Mean: {np.mean(data[-50:]):.2f} MB")
        print(f"  Total Processed: {np.sum(data):.2f} MB")

    if total_data is not None:
        print(f"\nSum (All UAVs):")
        print(f"  Min: {np.min(total_data):.2f} MB")
        print(f"  Max: {np.max(total_data):.2f} MB")
        print(f"  Mean: {np.mean(total_data):.2f} MB")
        print(f"  Std Dev: {np.std(total_data):.2f} MB")
        print(f"  Last 50 Episodes Mean: {np.mean(total_data[-50:]):.2f} MB")
        print(f"  Total Processed: {np.sum(total_data):.2f} MB")
    print(f"{'='*60}\n")

    plt.close()


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'results/MyEnv/UAV_EdgeComputing/rmappo/check')

    print(f"Searching for CSV files in: {base_dir}\n")

    # 检查目录是否存在
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return

    # 查找最新的 CSV 文件
    latest_csv = find_latest_csv(base_dir)

    if latest_csv is None:
        print("No episode_data_processed.csv files found.")
        return

    print(f"Latest CSV file: {latest_csv}\n")

    # 获取输出目录（与CSV同目录）
    output_dir = os.path.dirname(latest_csv)

    # 绘制处理数据
    plot_uav_processed_data(str(latest_csv), output_dir)


if __name__ == '__main__':
    main()
