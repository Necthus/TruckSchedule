from agent.reposition.base_agent import BaseRepositionAgent
from RLAlgo.reinforce import REINFORCEReposition
from simulator import Environment
from component import *
from parameter import *
from perception import PerceptionLayer, RepositionPerceptionRecord
from toolkit.time import print_with_time
import numpy as np


class DispatchAwareRLRepositionAgent(BaseRepositionAgent):
    """Dispatch感知的RL Reposition Agent：利用感知层反馈来优化厂站选择"""

    def __init__(self, train_mode=False):
        super().__init__(train_mode)
        self.algo = REINFORCEReposition(input_dim=7, hidden_dims=(64, 32))
        self.use_cost_reward = False  # 是否包含原有成本奖励，False则仅使用感知层奖励

        self.last_total_cost = None
        self.last_cost_reset_time = None

    @staticmethod
    def _compute_env_cost(env: Environment) -> float:
        """从Environment当前状态计算累计总成本"""
        go_fuel = env.go_distance * LOAD_GO_FUEL_CONSUMPTION_PER_KM * OIL_PRICE
        ret_fuel = env.return_distance * EMPTY_RETURN_FUEL_CONSUMPTION_PER_KM * OIL_PRICE
        idle_fuel = env.fuel_consumption * OIL_PRICE
        fuel_cost = go_fuel + ret_fuel + idle_fuel
        return fuel_cost + env.overtime_penalty + env.discontinuity_penalty

    def model_initialization(self, load_episode=-1):
        return self.algo.model_initialization(MODEL_REPOSITION_SAVE_DIR)

    def before_every_episode(self):
        self.algo.reset_transition()
        self.last_total_cost = None
        self.last_cost_reset_time = None

    def after_every_episode(self, env: Environment):
        if self.train_mode:
            current_total_cost = self._compute_env_cost(env)
            n_repositions = len(self.algo.transition['rewards'])

            if self.use_cost_reward:
                # 结算最后一次Reposition到当天结束的区间reward
                if n_repositions > 0 and self.last_cost_reset_time is not None and self.last_total_cost is not None:
                    delta_cost = current_total_cost - self.last_total_cost
                    end_time = env.current_time
                    time_interval_hours = (end_time - self.last_cost_reset_time).total_seconds() / 3600.0
                    if time_interval_hours > 0:
                        reward = -delta_cost / time_interval_hours
                        self.algo.transition['rewards'][-1] += reward

                # 厂站和工地加班补偿均匀分配给所有Reposition
                if n_repositions > 0:
                    overtime_total = env.station_overtime_cost + env.project_overtime_cost
                    overtime_per_repo = overtime_total / n_repositions
                    for i in range(n_repositions):
                        self.algo.transition['rewards'][i] -= overtime_per_repo

            total_reward = sum(self.algo.transition['rewards'])
            print_with_time(f'DispatchAwareRL Reposition cum reward = {total_reward}')
            self.algo.update()
            if (env.i_episode + 1) % SAVE_MODEL_FREQUENCY == 0:
                self.algo.save(MODEL_REPOSITION_SAVE_DIR, env.i_episode)
                print_with_time("DispatchAwareRL Reposition model saved")

    def select_reposition_station(self, current_pid, truck: Truck, available_sids, env: Environment):
        current_time = env.current_time
        current_total_cost = self._compute_env_cost(env)

        if self.train_mode and self.use_cost_reward:
            # 把上一个区间的成本增量reward回填到上一轮的占位slot
            if self.last_cost_reset_time is not None and self.last_total_cost is not None and len(self.algo.transition['rewards']) > 0:
                delta_cost = current_total_cost - self.last_total_cost
                time_interval_hours = (current_time - self.last_cost_reset_time).total_seconds() / 3600.0
                if time_interval_hours > 0:
                    reward = -delta_cost / time_interval_hours
                    self.algo.transition['rewards'][-1] += reward

        self.last_total_cost = current_total_cost
        self.last_cost_reset_time = current_time

        # 构建特征
        features_list = []
        for sid in available_sids:
            features = env.get_action_feature(sid, current_pid)
            features_list.append(features)

        features_array = np.array(features_list, dtype=np.float32)

        # RL选择动作
        force_greedy = not self.train_mode
        action_idx = self.algo.take_action(features_array, force_greedy=force_greedy)
        chosen_sid = available_sids[action_idx]

        # 记录state + action + reward占位（三列表始终对齐），后续由感知层回填
        if self.train_mode:
            self.algo.transition['states'].append(features_array)
            self.algo.transition['actions'].append(action_idx)
            self.algo.transition['rewards'].append(0.0)

        return chosen_sid

    def apply_perception_rewards(self, perception_layer: PerceptionLayer, env: Environment):
        """
        应用感知层的奖励信号到当前RL训练中。
        在dispatch发生后调用，将感知层计算的正负奖励追加到reposition的transition中。
        """
        if not self.train_mode:
            return

        if not perception_layer.dispatch_records:
            return

        last_dispatch = perception_layer.dispatch_records[-1]
        positive_rewards, negative_rewards = perception_layer.process_dispatch_result(
            env, last_dispatch)

        for repo_idx, reward in positive_rewards:
            if repo_idx < len(self.algo.transition['rewards']):
                self.algo.transition['rewards'][repo_idx] += reward

        for repo_idx, reward in negative_rewards:
            if repo_idx < len(self.algo.transition['rewards']):
                self.algo.transition['rewards'][repo_idx] += reward
