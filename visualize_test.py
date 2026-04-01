# -*- coding: utf-8 -*-
"""
UAV MEC 可视化测试脚本

用法：
  python visualize_test.py --model_dir results/MyEnv/UAV_EdgeComputing/rmappo/check/run1/models
  python visualize_test.py --model_dir <path> --uav0_pos 1,1,70 --uav1_pos 999,1,70
  python visualize_test.py --model_dir <path> --terminal_positions "250,250,1;750,250,1;250,750,1;750,750,1;500,250,1;500,750,1"
  python visualize_test.py --model_dir <path> --seed 42
"""

import sys
import os
import argparse
import numpy as np
import torch
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# ============================================================
# 默认位置配置
# ============================================================
DEFAULT_UAV_POSITIONS = [
    [1.0,   1.0, 70.0],   # UAV0 初始位置（ground_area=400）
    [399.0, 1.0, 70.0],   # UAV1 初始位置（ground_area=400）
]

# 对应 get_fixed_terminal_positions(6, ground_area=400)
# quarter=100, three_quarter=300, half=200
DEFAULT_TERMINAL_POSITIONS = [
    [100.0, 100.0, 1.0],  # 终端0：左下
    [300.0, 100.0, 1.0],  # 终端1：右下
    [100.0, 300.0, 1.0],  # 终端2：左上
    [300.0, 300.0, 1.0],  # 终端3：右上
    [200.0, 100.0, 1.0],  # 终端4：中下
    [200.0, 300.0, 1.0],  # 终端5：中上
]

MOVE_NAMES = {0:"悬停",1:"向上",2:"向下",3:"向左",4:"向右",5:"上升",6:"下降"}

# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="UAV MEC 可视化测试")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="模型目录，包含 actor.pt")
    parser.add_argument("--uav0_pos", type=str, default=None,
                        help="UAV0 初始位置，格式：x,y,z（默认：1,1,70）")
    parser.add_argument("--uav1_pos", type=str, default=None,
                        help="UAV1 初始位置，格式：x,y,z（默认：999,1,70）")
    parser.add_argument("--terminal_positions", type=str, default=None,
                        help="终端位置，格式：x1,y1,z1;x2,y2,z2;...（6个，分号分隔）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认：模型目录上级的 visualization/）")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="确定性策略")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最大步数（默认使用环境 episode_limit）")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="测试 episode 数量（默认：1）")
    return parser.parse_args()


def parse_pos(pos_str):
    parts = pos_str.strip().split(",")
    if len(parts) != 3:
        raise ValueError(f"位置格式错误: '{pos_str}'，应为 'x,y,z'")
    return [float(p) for p in parts]


def parse_terminal_positions(pos_str):
    return [parse_pos(t) for t in pos_str.strip().split(";")]


# ============================================================
# 模型加载
# ============================================================
def load_actor(model_dir, obs_space, action_space, device):
    from config import get_config
    from algorithms.algorithm.r_actor_critic import R_Actor
    parser = get_config()
    args = parser.parse_known_args([])[0]
    actor = R_Actor(args, obs_space, action_space, device)
    actor_path = os.path.join(model_dir, "actor.pt")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"找不到模型文件: {actor_path}")
    state_dict = torch.load(actor_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    print(f"[OK] 成功加载模型: {actor_path}")
    return actor


# ============================================================
# 环境包装器（注入自定义位置）
# ============================================================
class PatchedEnv:
    def __init__(self, uav_positions, terminal_positions):
        from envs.env_discrete import DiscreteActionEnv
        self.base_env = DiscreteActionEnv()
        self.uav_positions = np.array(uav_positions, dtype=np.float64)
        self.terminal_positions = terminal_positions
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.share_observation_space = self.base_env.share_observation_space
        self.num_agent = self.base_env.num_agent

    def reset(self):
        self.base_env.reset()
        core = self.base_env.env
        core.uav_positions = self.uav_positions.copy()
        for i, pos in enumerate(self.terminal_positions):
            if i < len(core.terminals):
                core.terminals[i]['position'] = np.array(pos, dtype=np.float64)
        obs = np.stack(core._get_obs())
        return obs

    def step(self, actions):
        return self.base_env.step(actions)

    def get_core(self):
        return self.base_env.env


# ============================================================
# 主测试循环
# ============================================================
def run_episode(actor, env, max_steps, deterministic, device):
    num_agents = env.num_agent
    hidden_size = 128
    recurrent_N = 1

    rnn_states = np.zeros((num_agents, recurrent_N, hidden_size), dtype=np.float32)
    masks = np.ones((num_agents, 1), dtype=np.float32)
    obs = env.reset()
    core = env.get_core()

    trajectory = []
    step_data = []

    for step in range(max_steps):
        uav_pos_snap = core.uav_positions.copy()
        uav_bat_snap = core.uav_battery.copy()

        # 推理
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(device)
            rnn_t = torch.from_numpy(rnn_states).float().to(device)
            mask_t = torch.from_numpy(masks).float().to(device)
            actions_list, new_rnn_list = [], []
            for aid in range(num_agents):
                act, _, new_rnn = actor(
                    obs_t[aid:aid+1], rnn_t[aid:aid+1], mask_t[aid:aid+1],
                    deterministic=deterministic,
                )
                actions_list.append(act.cpu().numpy()[0])
                new_rnn_list.append(new_rnn.cpu().numpy()[0])

        actions = np.stack(actions_list)
        rnn_states = np.stack(new_rnn_list)

        obs, rewards, dones, infos = env.step(actions)
        masks = np.ones((num_agents, 1), dtype=np.float32)
        for aid in range(num_agents):
            if dones[aid]:
                masks[aid] = 0.0
                rnn_states[aid] = np.zeros((recurrent_N, hidden_size), dtype=np.float32)

        trajectory.append({
            'step': step + 1,
            'uav_positions': uav_pos_snap,
            'uav_battery': uav_bat_snap,
            'actions': actions.copy(),
            'rewards': [float(r[0]) if hasattr(r, '__len__') else float(r) for r in rewards],
            'dones': list(dones),
            'infos': infos,
        })

        for aid in range(num_agents):
            info = infos[aid] if isinstance(infos, list) else infos
            served = info.get('selected_terminals', [])
            total_bits = info.get('processed_bits', 0.0)
            reward_val = float(rewards[aid][0]) if hasattr(rewards[aid], '__len__') else float(rewards[aid])
            n_served = len(served)

            if n_served > 0:
                bits_each = total_bits / n_served
                for term_id in served:
                    step_data.append({
                        'step': step + 1,
                        'uav_id': aid,
                        'terminal_id': term_id,
                        'processed_bits': round(bits_each, 2),
                        'processed_mb': round(bits_each / (8*1024*1024), 6),
                        'reward': round(reward_val, 6),
                        'battery_j': round(uav_bat_snap[aid], 2),
                        'uav_x': round(uav_pos_snap[aid, 0], 2),
                        'uav_y': round(uav_pos_snap[aid, 1], 2),
                        'uav_z': round(uav_pos_snap[aid, 2], 2),
                        'move_action': int(actions[aid, 0]),
                        'move_name': MOVE_NAMES.get(int(actions[aid, 0]), '?'),
                        'service_decision': int(actions[aid, 1]),
                        'num_to_serve': int(actions[aid, 2]),
                        'completion_ratio': round(info.get('task_completion_ratio', 0.0), 4),
                    })
            else:
                step_data.append({
                    'step': step + 1,
                    'uav_id': aid,
                    'terminal_id': -1,
                    'processed_bits': 0.0,
                    'processed_mb': 0.0,
                    'reward': round(reward_val, 6),
                    'battery_j': round(uav_bat_snap[aid], 2),
                    'uav_x': round(uav_pos_snap[aid, 0], 2),
                    'uav_y': round(uav_pos_snap[aid, 1], 2),
                    'uav_z': round(uav_pos_snap[aid, 2], 2),
                    'move_action': int(actions[aid, 0]),
                    'move_name': MOVE_NAMES.get(int(actions[aid, 0]), '?'),
                    'service_decision': int(actions[aid, 1]),
                    'num_to_serve': int(actions[aid, 2]),
                    'completion_ratio': round(info.get('task_completion_ratio', 0.0), 4),
                })

        if all(dones):
            print(f"\nEpisode 结束于第 {step + 1} 步")
            break

    return trajectory, step_data


# ============================================================
# CSV 输出
# ============================================================
def save_csv(step_data, output_path):
    if not step_data:
        print("[警告] 无数据可写入 CSV")
        return
    fieldnames = [
        'step', 'uav_id', 'terminal_id',
        'processed_bits', 'processed_mb',
        'reward', 'battery_j',
        'uav_x', 'uav_y', 'uav_z',
        'move_action', 'move_name', 'service_decision', 'num_to_serve',
        'completion_ratio',
    ]
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_data)
    print(f"[OK] CSV 已保存: {output_path}  ({len(step_data)} 行)")


# ============================================================
# 三维可视化
# ============================================================
def plot_3d_trajectory(trajectory, terminal_positions, output_path, ground_area=1000.0):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap  # noqa: F401
    except ImportError:
        print("[警告] matplotlib 未安装，跳过可视化。pip install matplotlib")
        return

    num_agents = len(trajectory[0]['uav_positions'])
    uav_colors  = ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b']
    term_colors = ['#f08030', '#6890f0', '#78c850', '#f8d030',
                   '#e8516a', '#98d8d8', '#705898']

    # 提取轨迹
    traj = {i: {'x':[], 'y':[], 'z':[]} for i in range(num_agents)}
    for frame in trajectory:
        for i in range(num_agents):
            traj[i]['x'].append(frame['uav_positions'][i, 0])
            traj[i]['y'].append(frame['uav_positions'][i, 1])
            traj[i]['z'].append(frame['uav_positions'][i, 2])

    # 计算各终端累计被处理量（用于气泡大小）
    term_bits = {i: 0.0 for i in range(len(terminal_positions))}
    for frame in trajectory:
        for aid, info in enumerate(frame['infos']):
            for tid in info.get('selected_terminals', []):
                n = len(info.get('selected_terminals', []))
                bits = info.get('processed_bits', 0.0) / n if n > 0 else 0
                term_bits[tid] = term_bits.get(tid, 0.0) + bits

    max_bits = max(term_bits.values()) if max(term_bits.values()) > 0 else 1.0

    # ---- 图1：三维轨迹主图 ----
    fig = plt.figure(figsize=(18, 8), facecolor='#1a1a2e')

    ax = fig.add_subplot(121, projection='3d')
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # 绘制地面网格
    for v in np.linspace(0, ground_area, 5):
        ax.plot([v, v], [0, ground_area], [0, 0], color='#2a2a4a', lw=0.5, alpha=0.5)
        ax.plot([0, ground_area], [v, v], [0, 0], color='#2a2a4a', lw=0.5, alpha=0.5)

    # 绘制终端（气泡大小=累计处理量）
    for idx, pos in enumerate(terminal_positions):
        ratio = term_bits.get(idx, 0.0) / max_bits
        size = 80 + ratio * 300
        color = term_colors[idx % len(term_colors)]
        ax.scatter(pos[0], pos[1], pos[2],
                   s=size, c=color, marker='^', zorder=5,
                   edgecolors='white', linewidths=0.8, alpha=0.9)
        ax.text(pos[0], pos[1], pos[2] + 18, f'T{idx}',
                color=color, fontsize=8, ha='center', fontweight='bold')

    # 绘制 UAV 轨迹
    for i in range(num_agents):
        c = uav_colors[i % len(uav_colors)]
        xs, ys, zs = traj[i]['x'], traj[i]['y'], traj[i]['z']
        n = len(xs)

        # 用颜色渐变表示时间（分段绘制）
        segs = max(1, n // 20)
        for s in range(0, n - 1, segs):
            e = min(s + segs + 1, n)
            alpha = 0.3 + 0.7 * (s / n)
            ax.plot(xs[s:e], ys[s:e], zs[s:e], color=c, lw=1.5, alpha=alpha)

        # 起点（实心圆）和终点（星形）
        ax.scatter(xs[0], ys[0], zs[0], s=120, c=c,
                   marker='o', zorder=10, edgecolors='white', linewidths=1.2)
        ax.scatter(xs[-1], ys[-1], zs[-1], s=200, c=c,
                   marker='*', zorder=10, edgecolors='white', linewidths=0.8)
        ax.text(xs[0], ys[0], zs[0] + 10, f'UAV{i} start',
                color=c, fontsize=7)
        ax.text(xs[-1], ys[-1], zs[-1] + 10, f'UAV{i} end',
                color=c, fontsize=7)

    ax.set_xlim(0, ground_area)
    ax.set_ylim(0, ground_area)
    ax.set_zlim(0, 130)
    ax.set_xlabel('X (m)', color='white', labelpad=6)
    ax.set_ylabel('Y (m)', color='white', labelpad=6)
    ax.set_zlabel('Z (m)', color='white', labelpad=6)
    ax.set_title('UAV 三维飞行轨迹', color='white', fontsize=13, pad=12)
    ax.view_init(elev=25, azim=-55)

    # 图例
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0],[0], color=uav_colors[i], lw=2, label=f'UAV {i}') for i in range(num_agents)]
    legend_elems += [Line2D([0],[0], marker='^', color='w', markerfacecolor=term_colors[j],
                             markersize=8, label=f'终端 {j}', lw=0) for j in range(len(terminal_positions))]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=7,
              facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

    # ---- 图2：电量和完成率曲线 ----
    ax2 = fig.add_subplot(222, facecolor='#16213e')
    steps = [f['step'] for f in trajectory]
    for i in range(num_agents):
        battery = [f['uav_battery'][i] / 324000.0 * 100 for f in trajectory]  # 转为百分比
        ax2.plot(steps, battery, color=uav_colors[i], lw=1.8, label=f'UAV {i} 电量')
    ax2.set_xlabel('Step', color='white')
    ax2.set_ylabel('电量 (%)', color='white')
    ax2.set_title('UAV 电量变化', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='gray', labelcolor='white')
    ax2.set_facecolor('#16213e')
    ax2.grid(True, alpha=0.2, color='gray')

    # ---- 图3：任务完成率曲线 ----
    ax3 = fig.add_subplot(224, facecolor='#16213e')
    completion = [f['infos'][0].get('task_completion_ratio', 0.0) * 100 for f in trajectory]
    ax3.fill_between(steps, completion, alpha=0.3, color='#51cf66')
    ax3.plot(steps, completion, color='#51cf66', lw=2)
    ax3.set_xlabel('Step', color='white')
    ax3.set_ylabel('完成率 (%)', color='white')
    ax3.set_title('任务完成率', color='white', fontsize=10)
    ax3.set_ylim(0, 105)
    ax3.tick_params(colors='white')
    ax3.set_facecolor('#16213e')
    ax3.grid(True, alpha=0.2, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"[OK] 可视化图已保存: {output_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 解析位置参数
    uav_positions = list(DEFAULT_UAV_POSITIONS)
    if args.uav0_pos:
        uav_positions[0] = parse_pos(args.uav0_pos)
        print(f"[配置] UAV0 位置: {uav_positions[0]}")
    if args.uav1_pos:
        uav_positions[1] = parse_pos(args.uav1_pos)
        print(f"[配置] UAV1 位置: {uav_positions[1]}")

    terminal_positions = list(DEFAULT_TERMINAL_POSITIONS)
    if args.terminal_positions:
        terminal_positions = parse_terminal_positions(args.terminal_positions)
        if len(terminal_positions) != 6:
            raise ValueError(f"需要6个终端位置，实际提供了 {len(terminal_positions)} 个")
        print(f"[配置] 使用自定义终端位置")
    else:
        print(f"[配置] 使用默认终端位置")

    # 输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.model_dir), 'visualization')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[配置] 输出目录: {output_dir}")

    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[配置] 使用设备: {device}")

    # 创建环境
    env = PatchedEnv(uav_positions, terminal_positions)
    max_steps = args.max_steps if args.max_steps else env.get_core().episode_limit
    print(f"[配置] 最大步数: {max_steps}")

    # 加载模型
    actor = load_actor(
        args.model_dir,
        env.observation_space[0],
        env.action_space[0],
        device
    )

    # 多 episode 循环
    print(f"\n开始测试，共 {args.num_episodes} 个 episode...")
    all_steps = []
    all_records = []

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep} ---")
        trajectory, step_data = run_episode(actor, env, max_steps, args.deterministic, device)
        print(f"  运行 {len(trajectory)} 步，{len(step_data)} 条服务记录")
        all_steps.append(len(trajectory))
        all_records.append(len(step_data))

        # 每个 episode 单独保存文件（按编号命名）
        ep_suffix = f"_ep{ep}" if args.num_episodes > 1 else ""

        csv_path = os.path.join(output_dir, f'step_service_data{ep_suffix}.csv')
        save_csv(step_data, csv_path)

        plot_path = os.path.join(output_dir, f'trajectory_3d{ep_suffix}.png')
        plot_3d_trajectory(trajectory, terminal_positions, plot_path,
                           ground_area=env.get_core().ground_area)

    # 汇总打印
    print(f"\n=== 测试完成 ===")
    print(f"  共测试 {args.num_episodes} 个 episode")
    print(f"  平均步数: {sum(all_steps)/len(all_steps):.1f}  "
          f"（最短 {min(all_steps)} 步，最长 {max(all_steps)} 步）")
    print(f"  输出目录: {output_dir}")


if __name__ == "__main__":
    main()
