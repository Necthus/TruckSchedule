from agent.reposition.base_agent import BaseRepositionAgent
from RLAlgo.reinforce import REINFORCEReposition
from simulator import Environment
from component import *
from parameter import *
from toolkit.time import print_with_time
import numpy as np


class RLRepositionAgent(BaseRepositionAgent):
    """基础RL Reposition Agent：根据厂站特征选择最优返回厂站"""

    def __init__(self, train_mode=False):
        super().__init__(train_mode)
        self.algo = REINFORCEReposition(input_dim=7, hidden_dims=(64, 32))

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
            print_with_time(f'RL Reposition cum reward = {total_reward}')
            self.algo.update()
            if (env.i_episode + 1) % SAVE_MODEL_FREQUENCY == 0:
                self.algo.save(MODEL_REPOSITION_SAVE_DIR, env.i_episode)
                print_with_time("RL Reposition model saved")

    def select_reposition_station(self, current_pid, truck: Truck, available_sids, env: Environment):
        current_time = env.current_time
        current_total_cost = self._compute_env_cost(env)

        if self.train_mode:
            # 把上一个区间的成本增量reward回填到上一轮的占位slot
            if self.last_cost_reset_time is not None and self.last_total_cost is not None and len(self.algo.transition['rewards']) > 0:
                delta_cost = current_total_cost - self.last_total_cost
                time_interval_hours = (current_time - self.last_cost_reset_time).total_seconds() / 3600.0
                if time_interval_hours > 0:
                    reward = -delta_cost / time_interval_hours
                    self.algo.transition['rewards'][-1] += reward

        self.last_total_cost = current_total_cost
        self.last_cost_reset_time = current_time

        # 构建每个可用厂站的特征
        features_list = []
        for sid in available_sids:
            features = env.get_action_feature(sid, current_pid)
            features_list.append(features)

        features_array = np.array(features_list, dtype=np.float32)

        # RL选择动作
        force_greedy = not self.train_mode
        action_idx = self.algo.take_action(features_array, force_greedy=force_greedy)

        # 记录state + action + reward占位（三列表始终对齐）
        if self.train_mode:
            self.algo.transition['states'].append(features_array)
            self.algo.transition['actions'].append(action_idx)
            self.algo.transition['rewards'].append(0.0)

        return available_sids[action_idx]
