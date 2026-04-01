import numpy as np

from .uav_comm_energy import RotorcraftParams
from .function import (
    calculate_horizontal_distance,       # 计算两点间水平距离
    calculate_propulsion_energy,         # 计算推进能耗
    calculate_velocity_from_positions,   # 从位置变化计算水平和垂直速度
    clip_position_to_boundary,           # 将位置裁剪到合法范围内
    generate_terminal_positions,         # 生成地面终端位置
    generate_uav_initial_positions,      # 生成UAV初始位置
    get_fixed_terminal_positions,        # 获取固定的终端基准位置
    initialize_all_terminal_tasks,       # 初始化所有终端任务
    update_all_terminals_progress,       # 更新所有终端任务进度
)


class EnvCore(object):
    """
    UAV搜索终端环境核心。

    动作空间（每个UAV单维离散）：
    - action[0]: 移动方向 (0-8)
        0=悬停, 1=上, 2=下, 3=左, 4=右
        5=左上, 6=左下, 7=右上, 8=右下
    """

    def __init__(self, use_discrete_action=False):
        # Multi-agent settings
        self.agent_num = 6
        self.num_terminals = 6
        self.action_dim = 1  # 只有8方向移动动作
        self.use_discrete_action = use_discrete_action  # 是否使用离散动作空间

        # Episode settings
        self.episode_limit = 4000
        self.time_slot = 1.0
        self.current_step = 0

        # Space and mobility settings
        self.ground_area = 2000.0
        self.height_min = 20.0
        self.height_max = 120.0
        self.initial_height = 70.0
        self.max_horizontal_speed = 10.0
        self.max_vertical_speed = 5.0
        self.delta_h = 5.0

        # Communication and computation settings
        self.transmit_power = 0.2  # 0.2 W (increased from 0.1)
        self.bandwidth = 5e6  # 5 MHz (increased from 1 MHz)
        self.carrier_frequency = 2.4e9
        self.noise_power_density = 4e-21
        self.antenna_gain = 2.0
        
        # LoS probability model parameters (for air-ground channel)
        self.los_a = 9.61  # LoS probability fitting parameter a (urban scenario)
        self.los_b = 0.16  # LoS probability fitting parameter b
        
        # LoS/NLoS additional path loss (dB)
        self.eta_los = 1.0   # LoS additional loss
        self.eta_nlos = 20.0 # NLoS additional loss
        
        self.cpu_freq_terminal = 1e9  # 2 GHz - Terminal local CPU (weak)
        self.cpu_freq_uav = 5e9  # 5 GHz - UAV CPU (medium)
        self.cpu_freq_ground = 1e9  # 100 GHz - Ground server CPU (strong)
        self.cpu_cycles_per_bit = 1000
        self.data_range = (200000.0, 300000.0)  # KB = 200-300 MB (increased from 100-200 KB)从bit上来看是 1-2e9 bits

        # Energy settings
        self.rotor_params = RotorcraftParams()
        self.battery_capacity = 800000.0  # 800 kJ
        self.energy_per_bit = 1e-6  # 1 μJ/bit (increased from 1 nJ/bit)

        # Reward settings
        self.reward_per_bit = 0  # 每bit数据的奖励参数（最开始是5e-7）
        self.energy_penalty = 10e-4  # 能耗惩罚系数
        self.completion_bonus_half = 15.0  # 完成50%的奖励
        self.completion_bonus_full = 30.0  # 完成100%的奖励
        self.service_reward = 0.1  # 提供卸载服务的小奖励
        self.invalid_service_penalty = 0.5  # 服务已完成终端的惩罚
        self.battery_depleted_penalty = 50.0  # 电池耗尽的惩罚
        self.timeout_penalty = 60.0  # 超时未完成所有任务的惩罚
        self.discovery_radius = 200.0 * np.sqrt(2)  # ≈ 282.8m，等于格子对角线长度
        self.discovery_reward = 100.0  # 首次发现终端奖励（主要奖励）
        self.undiscovered_obs_placeholder = 1e6  # 未发现终端观测占位值

        # 覆盖格子奖励设置
        # 格子边长 = 200m，对角线 = 200*sqrt(2) ≈ 282.8m = 发现半径
        # UAV进入格子必能发现格子内所有终端
        self.grid_size = 200.0  # 格子边长 200m
        self.grid_cols = int(np.ceil(self.ground_area / self.grid_size))  # 10列
        self.grid_rows = int(np.ceil(self.ground_area / self.grid_size))  # 10行
        self.coverage_reward = 50.0  # 每覆盖一个新格子的奖励（辅助探索，约为发现奖励的1/10）

        # Observation: 10x10 grid map flattened to 100-dim vector (0=uncovered, 1=covered, 2=self position)
        self.obs_shape = (self.grid_rows * self.grid_cols,)
        self.obs_dim = self.grid_rows * self.grid_cols

        # Runtime state
        self.uav_positions = None
        self.uav_battery = None
        self.uav_processing_data = None
        self.terminals = None
        self.terminal_discovered = None
        self.episode_count = -1  # episode 计数器（从-1开始，reset后第一个episode为0，与runner对齐）
        self.padding_mode = False  # 提前达成终止条件后，剩余step仅计数不执行动作
        self.padding_end_reason = None

    def _build_terminal_states(self):
        # 获取固定的终端基准位置（从 function.py）
        base_positions = get_fixed_terminal_positions(
            num_terminals=self.num_terminals,
            ground_area=self.ground_area
        )
        
        # 生成终端位置（基准位置 + 较大随机偏移）
        terminal_positions = generate_terminal_positions(
            num_terminals=self.num_terminals,
            ground_area=self.ground_area,
            base_positions=base_positions,
            variance=300.0,  # 在基准位置 ±300m 范围内随机偏移
        )
        
        terminal_tasks = initialize_all_terminal_tasks(
            num_terminals=self.num_terminals,
            data_range=self.data_range,
            cpu_cycles_per_bit=self.cpu_cycles_per_bit,
        )
        for term_id, task in enumerate(terminal_tasks):
            task["position"] = terminal_positions[term_id]
        return terminal_tasks

    def _build_grid_observation(self, uav_id: int) -> np.ndarray:
        """构建格子观测向量: 0=未覆盖, 1=已覆盖, 2=本UAV位置"""
        grid_obs = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        # 已覆盖格子标记为1
        for (row, col) in self.global_covered_grids.keys():
            grid_obs[row, col] = 1.0

        # 本UAV当前位置标记为2
        pos = self.uav_positions[uav_id]
        col = min(int(pos[0] / self.grid_size), self.grid_cols - 1)
        row = min(int(pos[1] / self.grid_size), self.grid_rows - 1)
        grid_obs[row, col] = 2.0

        # 展平为100维向量，后续在网络里恢复为10x10做CNN
        return grid_obs.reshape(-1)

    def _get_obs(self):
        obs = []
        for uav_id in range(self.agent_num):
            one_obs = self._build_grid_observation(uav_id)
            obs.append(one_obs)
        return obs

    def reset(self):
        self.current_step = 0
        self.padding_mode = False
        self.padding_end_reason = None
        self.episode_count += 1  # 每次 reset 时 episode 编号自增
        self.uav_positions = generate_uav_initial_positions(
            num_uavs=self.agent_num,
            ground_area=self.ground_area,
            initial_height=self.initial_height,
            mode="grid",
        )
        # 统一将所有UAV初始位置设为(1,1,70)
        self.uav_positions[:] = np.array([1.0, 1.0, self.initial_height])
        self.uav_battery = np.full(self.agent_num, self.battery_capacity, dtype=np.float64)
        self.uav_processing_data = {uav_id: 0.0 for uav_id in range(self.agent_num)}
        self.terminals = self._build_terminal_states()
        self.terminal_discovered = np.zeros(self.num_terminals, dtype=bool)
        # 追踪每个终端被发现的步数（-1表示未被发现）
        self.terminal_discovery_step = {
            term_id: -1 for term_id in range(self.num_terminals)
        }
        # 追踪每个终端的完成进度（用于奖励发放）
        self.terminal_completion_milestones = {
            term_id: {'half': False, 'full': False} 
            for term_id in range(self.num_terminals)
        }
        # 追踪UAV是否电池耗尽（用于停止行动）
        self.uav_depleted = np.zeros(self.agent_num, dtype=bool)
        
        # 初始化每个UAV的格子覆盖记录（全局共享，记录哪个UAV首先覆盖了该格子）
        # key: (row, col)，value: 首次覆盖该格子的uav_id
        self.global_covered_grids = {}
        
        # 输出初始位置信息
        #print("\n" + "=" * 50)
        #print("=== Episode 开始 - 初始位置 ===")
        #print("UAV位置:")
        #for uav_id in range(self.agent_num):
        #    pos = self.uav_positions[uav_id]
        #    print(f"  UAV{uav_id}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
        
        #print("\n终端位置:")
        #for term_id, terminal in enumerate(self.terminals):
        #    pos = terminal['position']
        #    data_mb = terminal['total_data_bits'] / (1024 * 8 * 1000)  # bits -> MB
        #    print(f"  终端{term_id}: x={pos[0]:.2f}, y={pos[1]:.2f}, 数据量={data_mb:.2f} MB")
        #print("=" * 50 + "\n")
        
        return self._get_obs()

    def step(self, actions):
        # 根据动作空间类型处理actions
        if self.use_discrete_action:
            # 离散动作：直接使用整数
            actions = np.asarray(actions, dtype=np.int32)
        else:
            # 连续动作：转换为float32
            actions = np.asarray(actions, dtype=np.float32)

        if actions.ndim == 1:
            actions = actions.reshape(self.agent_num, -1)

        # 若提前达成终止条件，后续step进入占位模式：仅计数，不执行任何动作
        if self.padding_mode:
            self.current_step += 1
            timeout = self.current_step >= self.episode_limit
            obs = self._get_obs()
            rewards = [[0.0] for _ in range(self.agent_num)]
            dones = [bool(timeout) for _ in range(self.agent_num)]
            discovered_count = int(np.sum(self.terminal_discovered))
            discovery_ratio = discovered_count / self.num_terminals if self.num_terminals > 0 else 1.0
            infos = []
            for uav_id in range(self.agent_num):
                infos.append({
                    "selected_terminals": [],
                    "service_decision": False,
                    "num_terminals_to_serve": 0,
                    "num_served_terminals": 0,
                    "processed_bits": 0.0,
                    "propulsion_energy_j": 0.0,
                    "computation_energy_j": 0.0,
                    "communication_energy_j": 0.0,
                    "total_energy_j": 0.0,
                    "battery": float(self.uav_battery[uav_id]),
                    "num_invalid_services": 0,
                    "uav_depleted": bool(self.uav_depleted[uav_id]),
                    "newly_discovered_terminals": [],
                    "num_discovered_terminals": discovered_count,
                    "all_tasks_completed": bool(discovered_count == self.num_terminals),
                    "task_completion_ratio": float(discovery_ratio),
                    "episode_step": int(self.current_step),
                    "out_of_battery": bool(np.any(self.uav_depleted)),
                    "timeout": bool(timeout),
                    "all_terminals_discovered": bool(discovered_count == self.num_terminals),
                    "terminal_discovery_ratio": float(discovery_ratio),
                    "padding_mode": True,
                    "padding_reason": self.padding_end_reason,
                })
            return [obs, rewards, dones, infos]

        rewards = []
        dones = []
        infos = []

        uav_terminal_progress = {uav_id: {} for uav_id in range(self.agent_num)}
        selected_terminals = {}

        for uav_id in range(self.agent_num):
            # 检查UAV是否电池耗尽，如果耗尽则跳过所有行动
            if self.uav_depleted[uav_id]:  #电量二值化变量
                # 电池耗尽的UAV保持原位，不执行任何动作
                rewards.append([0.0])
                dones.append(False)
                infos.append({
                    "selected_terminals": [],
                    "service_decision": False,
                    "num_terminals_to_serve": 0,
                    "num_served_terminals": 0,
                    "processed_bits": 0.0,
                    "propulsion_energy_j": 0.0,
                    "computation_energy_j": 0.0,
                    "communication_energy_j": 0.0,
                    "total_energy_j": 0.0,
                    "battery": 0.0,
                    "num_invalid_services": 0,
                    "uav_depleted": True,
                    "newly_discovered_terminals": [],
                    "num_discovered_terminals": int(np.sum(self.terminal_discovered)),
                })
                continue

            action_vec = actions[uav_id]  # 拿出当前单个无人机动作向量

            # 8方向移动动作解码
            # action_vec[0]: 0=悬停,1=上,2=下,3=左,4=右,5=左上,6=左下,7=右上,8=右下（注：index从0开始共9格，但只用8方向+悬停）
            if self.use_discrete_action:
                move_dir = int(action_vec[0]) if hasattr(action_vec, '__len__') else int(action_vec)
            else:
                # 连续动作映射到9个方向（0-8）
                val = float(action_vec[0]) if hasattr(action_vec, '__len__') else float(action_vec)
                move_dir = int(np.clip(round((val + 1) / 2 * 8), 0, 8))

            # 解码为水平移动的dx, dy
            _dir_map = {
                0: (0, 0),    # 悬停
                1: (0, 1),    # 上
                2: (0, -1),   # 下
                3: (-1, 0),   # 左
                4: (1, 0),    # 右
                5: (-1, 1),   # 左上
                6: (-1, -1),  # 左下
                7: (1, 1),    # 右上
                8: (1, -1),   # 右下
            }
            dx, dy = _dir_map.get(move_dir, (0, 0))
            service_decision = False
            num_terminals_to_serve = 0


            old_pos = self.uav_positions[uav_id].copy()
            # 计算移动后的位置（8方向 + 悬停，固定高度）
            step_dist = self.max_horizontal_speed * self.time_slot
            # 斜向移动归一化，保持速度一致
            if dx != 0 and dy != 0:
                norm = step_dist / np.sqrt(2)
            else:
                norm = step_dist
            moved_pos = old_pos.copy()
            moved_pos[0] += dx * norm
            moved_pos[1] += dy * norm
            # 固定高度：UAV在搜索阶段保持初始高度，忽略垂直动作
            moved_pos[2] = self.initial_height
            moved_pos = clip_position_to_boundary(
                moved_pos, self.ground_area, self.height_min, self.height_max
            )
            self.uav_positions[uav_id] = moved_pos

            # 覆盖格子奖励：进入未曾覆盖的格子获得奖励，进入已被其他UAV覆盖的格子受到惩罚
            grid_col = int(moved_pos[0] / self.grid_size)
            grid_row = int(moved_pos[1] / self.grid_size)
            grid_col = min(grid_col, self.grid_cols - 1)
            grid_row = min(grid_row, self.grid_rows - 1)
            grid_key = (grid_row, grid_col)
            if grid_key not in self.global_covered_grids:
                # 全局首次覆盖：获得奖励，记录覆盖者
                self.global_covered_grids[grid_key] = uav_id
                coverage_bonus = self.coverage_reward
            elif self.global_covered_grids[grid_key] != uav_id:
                # 其他UAV已覆盖过：受到惩罚（重复区域）
                coverage_bonus = -self.coverage_reward * 0.001
            else:
                # 自己之前覆盖过：无奖励无惩罚
                coverage_bonus = 0.0

            # 发现机制：首次进入终端200m范围内触发一次奖励
            newly_discovered_terminals = []
            for terminal_id in range(self.num_terminals):
                if self.terminal_discovered[terminal_id]:
                    continue
                terminal_pos = self.terminals[terminal_id]["position"]
                distance = calculate_horizontal_distance(moved_pos, terminal_pos)
                if distance <= self.discovery_radius:
                    self.terminal_discovered[terminal_id] = True
                    self.terminal_discovery_step[terminal_id] = self.current_step
                    newly_discovered_terminals.append(terminal_id)

            v_horizontal, v_vertical = calculate_velocity_from_positions(
                pos_old=old_pos,
                pos_new=moved_pos,
                time_slot=self.time_slot,
            )


            # 核心逻辑：搜索阶段无服务动作
            total_processed_bits = 0.0
            total_communication_energy = 0.0
            total_computation_energy = 0.0
            num_invalid_services = 0
            served_terminal_ids = []
            selected_terminals[uav_id] = served_terminal_ids
            self.uav_processing_data[uav_id] += total_processed_bits

            # 计算推进能耗
            propulsion_energy, _ = calculate_propulsion_energy(
                v_horizontal=v_horizontal,
                v_vertical=v_vertical,
                time_slot=self.time_slot,
                rotor_params=self.rotor_params,
            )
            total_energy = propulsion_energy

            # 更新电池电量
            self.uav_battery[uav_id] = max(0.0, self.uav_battery[uav_id] - total_energy)
            
            # 检查电池是否耗尽
            if self.uav_battery[uav_id] <= 0.0:
                self.uav_depleted[uav_id] = True

            # 计算奖励
            # 正奖励：处理数据量
            reward = total_processed_bits * self.reward_per_bit
            # 正奖励：覆盖新格子
            reward += coverage_bonus
            # 正奖励：首次发现终端
            reward += self.discovery_reward * len(newly_discovered_terminals)
                        
            # 正奖励：提供卸载服务（只要服务了就有小奖励）
            #if len(served_terminal_ids) > 0:
               # reward += self.service_reward * len(served_terminal_ids)
            
            # 负奖励：能耗
            reward -= total_energy * self.energy_penalty
            # 惩罚：服务已完成的终端
            if num_invalid_services > 0:
                reward -= self.invalid_service_penalty * num_invalid_services

            rewards.append([float(reward)])
            infos.append(
                {
                    "selected_terminals": served_terminal_ids,
                    "service_decision": bool(service_decision),
                    "num_terminals_to_serve": int(num_terminals_to_serve),
                    "num_served_terminals": len(served_terminal_ids),
                    "processed_bits": float(total_processed_bits),
                    "propulsion_energy_j": float(propulsion_energy),
                    "computation_energy_j": float(total_computation_energy),
                    "communication_energy_j": float(total_communication_energy),
                    "total_energy_j": float(total_energy),
                    "battery": float(self.uav_battery[uav_id]),
                    "num_invalid_services": int(num_invalid_services),
                    "uav_depleted": bool(self.uav_depleted[uav_id]),
                    "newly_discovered_terminals": newly_discovered_terminals,
                    "num_discovered_terminals": int(np.sum(self.terminal_discovered)),
                }
            )

        self.terminals, completed_terminal_ids = update_all_terminals_progress(
            terminals=self.terminals,
            uav_terminal_progress=uav_terminal_progress,
            cpu_freq_terminal=self.cpu_freq_terminal,
            cpu_cycles_per_bit=self.cpu_cycles_per_bit,
            time_slot=self.time_slot,
        )
        
        # 计算完成奖励：两段式奖励（50%和100%）
        # 使用统一的milestone追踪系统
        '''
        for uav_id in range(self.agent_num):
            # 跳过电池耗尽的UAV
            if self.uav_depleted[uav_id]:
                continue
                
            served_terminal_ids = selected_terminals[uav_id]
            for term_id in served_terminal_ids:
                served_bits = uav_terminal_progress[uav_id].get(term_id, 0.0)
                if served_bits > 0.0:
                    terminal = self.terminals[term_id]
                    total_bits = terminal['total_data_bits']
                    remaining_bits = terminal['remaining_data_bits']
                    completion_ratio = 1.0 - (remaining_bits / total_bits)
                    
                    # 50%完成奖励
                    if completion_ratio >= 0.5 and not self.terminal_completion_milestones[term_id]['half']:
                        rewards[uav_id][0] += self.completion_bonus_half
                        self.terminal_completion_milestones[term_id]['half'] = True
                    
                    # 100%完成奖励
                    if remaining_bits <= 0 and not self.terminal_completion_milestones[term_id]['full']:
                        rewards[uav_id][0] += self.completion_bonus_full
                        self.terminal_completion_milestones[term_id]['full'] = True
'''
        self.current_step += 1
        discovered_count = int(np.sum(self.terminal_discovered))
        discovery_ratio = discovered_count / self.num_terminals if self.num_terminals > 0 else 1.0
        all_discovered = discovered_count == self.num_terminals
        out_of_battery = bool(np.any(self.uav_depleted))
        timeout = self.current_step >= self.episode_limit

        # 提前终止条件进入padding模式，直到episode_limit才真正done
        if not self.padding_mode and not timeout and (all_discovered or out_of_battery):
            self.padding_mode = True
            self.padding_end_reason = "all_discovered" if all_discovered else "out_of_battery"
            early_reason = "终端全部发现" if all_discovered else "电量耗尽"
            print("\n=== Episode 提前结束（进入padding）===")
            print(f"Episode: {self.episode_count}  |  当前Step: {self.current_step} / {self.episode_limit}  |  原因: {early_reason}")
            print(f"发现终端数: {discovered_count} / {self.num_terminals}")
            print("后续step将仅占位计数，不执行动作，直到episode_limit后切换下一episode。")
            print("=" * 42 + "\n")

        episode_end = bool(timeout)
        # 添加终局惩罚      
        for uav_id in range(self.agent_num):
            # 超时未完成发现目标的惩罚
            if timeout and not all_discovered:
                rewards[uav_id][0] -= self.timeout_penalty

            dones.append(bool(episode_end))
            infos[uav_id]["all_tasks_completed"] = bool(all_discovered)
            infos[uav_id]["task_completion_ratio"] = float(discovery_ratio)
            infos[uav_id]["episode_step"] = int(self.current_step)
            infos[uav_id]["out_of_battery"] = bool(np.any(self.uav_depleted))
            infos[uav_id]["timeout"] = bool(timeout)
            infos[uav_id]["all_terminals_discovered"] = bool(all_discovered)
            infos[uav_id]["terminal_discovery_ratio"] = float(discovery_ratio)
            infos[uav_id]["num_discovered_terminals"] = int(np.sum(self.terminal_discovered))
            infos[uav_id]["padding_mode"] = bool(self.padding_mode)
            infos[uav_id]["padding_reason"] = self.padding_end_reason

        # Episode真正结束时输出统计信息
        if episode_end:
            # 判断结束原因
            if all_discovered:
                end_reason = "终端全部发现"
            elif out_of_battery:
                end_reason = "电量耗尽"
            else:
                end_reason = "超时"
            print("\n=== Episode 结束统计 ===")
            print(f"Episode: {self.episode_count}  |  结束Step: {self.current_step} / {self.episode_limit}  |  原因: {end_reason}")
            print(f"发现终端数: {discovered_count} / {self.num_terminals}")
            print("终端发现情况:")
            for term_id in range(self.num_terminals):
                pos = self.terminals[term_id]['position']
                step = self.terminal_discovery_step[term_id]
                if step >= 0:
                    print(f"  终端{term_id} (x={pos[0]:.0f}, y={pos[1]:.0f}): 第 {step} 步被发现")
                else:
                    print(f"  终端{term_id} (x={pos[0]:.0f}, y={pos[1]:.0f}): 未被发现")
            print("=" * 30 + "\n")

        obs = self._get_obs()
        return [obs, rewards, dones, infos]
