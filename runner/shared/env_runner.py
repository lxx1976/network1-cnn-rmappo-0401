"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import numpy as np
import torch
import csv
import os
from runner.shared.base_runner import Runner

# import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

        # 初始化episode数据统计
        self.episode_data_stats = []
        self.csv_path = os.path.join(str(self.run_dir), 'episode_stats.csv')

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # 初始化episode数据收集
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            episode_total_rewards = 0.0
            episode_discovered_terminals = 0

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,#价值估计
                    actions,#动作
                    action_log_probs,#动作在当前策略下的对数概率
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # 收集每个step的奖励和发现终端数
                episode_total_rewards += np.mean(rewards)
                for env_id in range(self.n_rollout_threads):
                    # 取最后一个agent的info（所有agent的num_discovered_terminals相同）
                    if 'num_discovered_terminals' in infos[env_id][0]:
                        episode_discovered_terminals = infos[env_id][0]['num_discovered_terminals']

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # 保存episode统计数据
            episode_stat = {
                'episode': episode,
                'total_steps': (episode + 1) * self.episode_length * self.n_rollout_threads,
                'average_reward': float(episode_total_rewards / self.episode_length),
                'num_discovered_terminals': int(episode_discovered_terminals),
            }

            self.episode_data_stats.append(episode_stat)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                # if self.env_name == "MPE":
                #     env_infos = {}
                #     for agent_id in range(self.num_agents):
                #         idv_rews = []
                #         for info in infos:
                #             if 'individual_reward' in info[agent_id].keys():
                #                 idv_rews.append(info[agent_id]['individual_reward'])
                #         agent_k = 'agent%i/individual_rewards' % agent_id
                #         env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                # self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        # 训练结束后保存CSV和生成可视化
        self.save_episode_data_to_csv()
        self.visualize_episode_data()

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, ...]

        # 将多维观测展平为向量，兼容CNN输入前的buffer存储
        obs = obs.reshape(self.n_rollout_threads, self.num_agents, -1).astype(np.float32)

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)  # shape = [env_num, agent_num * obs_dim]
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1
            )  # shape = [env_num, agent_num, agent_num * obs_dim]
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))  # [env_num, agent_num, action_dim]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )  # [env_num, agent_num, 1]
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            # 对于MultiDiscrete动作空间，直接使用离散动作，不需要one-hot编码
            # For MultiDiscrete action space, use discrete actions directly without one-hot encoding
            actions_env = actions
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            # actions  --> actions_env : shape:[10, 1] --> [5, 2, 5]
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            # TODO 这里改造成自己环境需要的形式即可
            # TODO Here, you can change the shape of actions_env to fit your environment
            actions_env = actions
            # raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # 将多维观测展平为向量，保持与buffer定义一致
        obs = obs.reshape(self.n_rollout_threads, self.num_agents, -1).astype(np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                # 对于MultiDiscrete动作空间，直接使用离散动作，不需要one-hot编码
                eval_actions_env = eval_actions
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    # 对于MultiDiscrete动作空间，直接使用离散动作，不需要one-hot编码
                    actions_env = actions
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

    def save_episode_data_to_csv(self):
        """保存episode数据处理统计到CSV文件"""
        if len(self.episode_data_stats) == 0:
            print("No episode data to save.")
            return

        print(f"\n保存episode数据统计到: {self.csv_path}")

        # 获取所有列名
        fieldnames = list(self.episode_data_stats[0].keys())

        # 写入CSV
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.episode_data_stats)

        print(f"成功保存 {len(self.episode_data_stats)} 个episode的数据统计")

    def visualize_episode_data(self):
        """生成奖励和终端发现数的可视化图表"""
        if len(self.episode_data_stats) == 0:
            print("No episode data to visualize.")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib未安装，跳过可视化。请运行: pip install matplotlib")
            return

        print("\n生成奖励和终端发现数可视化图表...")

        # 提取数据
        episodes = [stat['episode'] for stat in self.episode_data_stats]
        avg_rewards = [stat['average_reward'] for stat in self.episode_data_stats]
        discovered = [stat['num_discovered_terminals'] for stat in self.episode_data_stats]

        # 平滑曲线（滑动平均，窗口=10）
        def moving_average(data, window=10):
            if len(data) < window:
                return data
            result = []
            for i in range(len(data)):
                start = max(0, i - window + 1)
                result.append(np.mean(data[start:i+1]))
            return result

        smoothed_rewards = moving_average(avg_rewards)
        smoothed_discovered = moving_average(discovered)

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 图1: 每episode平均奖励
        ax1.plot(episodes, avg_rewards, color='#aec6cf', linewidth=1, alpha=0.5, label='Raw')
        ax1.plot(episodes, smoothed_rewards, color='#2166ac', linewidth=2.5, label='Smoothed (window=10)')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Average Step Reward', fontsize=12)
        ax1.set_title('Average Reward per Episode', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        # 图2: 每episode发现的终端数
        num_terminals = self.episode_data_stats[0].get('num_discovered_terminals', 0)
        ax2.bar(episodes, discovered, color='#91cf60', alpha=0.6, label='Discovered terminals')
        ax2.plot(episodes, smoothed_discovered, color='#1a9641', linewidth=2.5, label='Smoothed (window=10)')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Terminals Discovered', fontsize=12)
        ax2.set_title('Number of Terminals Discovered per Episode', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 7)
        ax2.set_yticks(range(0, 7))
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(fontsize=11)

        plt.tight_layout()

        # 保存图表
        plot_path = os.path.join(str(self.run_dir), 'episode_training_stats.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"可视化图表已保存到: {plot_path}")
