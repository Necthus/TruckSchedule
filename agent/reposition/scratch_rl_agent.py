import json
import os

import numpy as np

from agent.reposition.base_agent import BaseRepositionAgent
from component import Truck
from parameter import (
    EMPTY_RETURN_FUEL_CONSUMPTION_PER_KM,
    LOAD_GO_FUEL_CONSUMPTION_PER_KM,
    MODEL_REPOSITION_SCRATCH_COMBINED_BEST_EPOCH,
    MODEL_REPOSITION_SCRATCH_COMBINED_SAVE_DIR,
    MODEL_REPOSITION_SCRATCH_COST_ONLY_BEST_EPOCH,
    MODEL_REPOSITION_SCRATCH_COST_ONLY_SAVE_DIR,
    MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_COST_BEST_EPOCH,
    MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_COST_SAVE_DIR,
    MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_SHAPING_BEST_EPOCH,
    MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_SHAPING_SAVE_DIR,
    MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_SAVE_DIR,
    MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_BEST_EPOCH,
    MODEL_REPOSITION_SCRATCH_PERCEPTION_ONLY_BEST_EPOCH,
    MODEL_REPOSITION_SCRATCH_PERCEPTION_ONLY_SAVE_DIR,
    MODEL_REPOSITION_SCRATCH_RL_BEST_EPOCH,
    MODEL_REPOSITION_SCRATCH_RL_SAVE_DIR,
    OIL_PRICE,
    SAVE_MODEL_FREQUENCY,
    SCRATCH_COMBINED_PERCEPTION_REWARD_SCALE,
)
from perception import PerceptionLayer
from RLAlgo.scratch_policy_gradient import ScratchPolicyGradientReposition
from simulator import Environment
from toolkit.time import print_with_time


class ScratchRLRepositionAgent(BaseRepositionAgent):
    """从随机初始化训练的 Reposition RL，不使用教师策略或预训练权重。"""

    input_dim = 12
    model_save_dir = MODEL_REPOSITION_SCRATCH_RL_SAVE_DIR
    best_model_epoch = MODEL_REPOSITION_SCRATCH_RL_BEST_EPOCH
    cost_reward_scale = 1.0 / 10000.0
    shaped_reward_scale = 1.0
    use_cost_reward = True
    use_shaping_reward = True
    use_perception_reward = False

    def __init__(self, train_mode=False):
        super().__init__(train_mode)
        self.algo = ScratchPolicyGradientReposition(input_dim=self.input_dim)
        self.last_total_cost = None
        self.last_cost_reset_time = None
        self._reset_reward_stats()

    def _reset_reward_stats(self):
        self.reward_stats = {
            'shape': 0.0,
            'cost_delta': 0.0,
            'overtime': 0.0,
            'perception_raw': 0.0,
            'perception_scaled': 0.0,
            'reposition_count': 0,
            'perception_event_count': 0,
        }

    def _add_reward_stat(self, key, value):
        self.reward_stats[key] = self.reward_stats.get(key, 0.0) + value

    @staticmethod
    def _compute_env_cost(env: Environment) -> float:
        go_fuel = env.go_distance * LOAD_GO_FUEL_CONSUMPTION_PER_KM * OIL_PRICE
        ret_fuel = env.return_distance * EMPTY_RETURN_FUEL_CONSUMPTION_PER_KM * OIL_PRICE
        idle_fuel = env.fuel_consumption * OIL_PRICE
        return go_fuel + ret_fuel + idle_fuel + env.overtime_penalty + env.discontinuity_penalty

    @staticmethod
    def _raw_action_feature(sid, current_pid, env: Environment):
        return env.get_action_feature(sid, current_pid)

    @classmethod
    def _build_feature(cls, sid, current_pid, env: Environment):
        dispatch_count, truck_num, distance, next_return, returning_count, future_30, future_60, future_120 = (
            cls._raw_action_feature(sid, current_pid, env)
        )
        future_need = future_30 + 0.5 * future_60 + 0.25 * future_120
        supply = truck_num + returning_count
        shortage = future_need - supply
        shortage_ratio = shortage / (future_need + supply + 1.0)

        return [
            dispatch_count / 10.0,
            truck_num / 20.0,
            distance / 30.0,
            next_return / 240.0,
            returning_count / 20.0,
            future_30 / 5.0,
            future_60 / 10.0,
            future_120 / 20.0,
            future_need / 10.0,
            supply / 20.0,
            shortage / 10.0,
            shortage_ratio,
        ]

    @classmethod
    def _action_shaping_reward(cls, sid, current_pid, env: Environment):
        dispatch_count, truck_num, distance, next_return, returning_count, future_30, future_60, future_120 = (
            cls._raw_action_feature(sid, current_pid, env)
        )
        future_need = future_30 + 0.5 * future_60 + 0.25 * future_120
        supply = truck_num + returning_count
        shortage_before = max(0.0, future_need - supply)
        shortage_after = max(0.0, future_need - supply - 1.0)
        shortage_reduction = shortage_before - shortage_after
        low_stock_bonus = max(0.0, 8.0 - supply)

        return (
            1.20 * low_stock_bonus
            + 4.00 * shortage_reduction
            + 0.45 * future_need
            - 0.90 * truck_num
            - 0.45 * returning_count
            - 0.08 * distance
            - 0.01 * next_return
            - 0.03 * dispatch_count
        )

    def model_initialization(self, load_episode=-1):
        if load_episode < 0 and not self.train_mode:
            load_episode = self._load_recorded_best_epoch()
            if load_episode is None:
                load_episode = self.best_model_epoch
        return self.algo.model_initialization(self.model_save_dir, load_episode=load_episode)

    def _load_recorded_best_epoch(self):
        metadata_path = os.path.join(self.model_save_dir, 'best_checkpoint.json')
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        best_epoch = int(metadata['best_epoch'])
        print_with_time(
            f"从{metadata_path}读取Scratch best checkpoint: "
            f"epoch={best_epoch}, validation_cost={metadata.get('validation_cost')}"
        )
        return best_epoch

    def before_every_episode(self):
        self.algo.reset_transition()
        self.last_total_cost = None
        self.last_cost_reset_time = None
        self._reset_reward_stats()

    def _assign_previous_cost_reward(self, env: Environment):
        if not self.train_mode:
            return
        if self.last_total_cost is None or self.last_cost_reset_time is None:
            return
        if not self.use_cost_reward:
            return
        if not self.algo.transition['rewards']:
            return

        current_total_cost = self._compute_env_cost(env)
        hours = (env.current_time - self.last_cost_reset_time).total_seconds() / 3600.0
        if hours > 0:
            delta_cost = current_total_cost - self.last_total_cost
            reward = -delta_cost * self.cost_reward_scale / hours
            self.algo.transition['rewards'][-1] += reward
            self._add_reward_stat('cost_delta', reward)

    def select_reposition_station(self, current_pid, truck: Truck, available_sids, env: Environment):
        self._assign_previous_cost_reward(env)
        self.last_total_cost = self._compute_env_cost(env)
        self.last_cost_reset_time = env.current_time

        features = np.array(
            [self._build_feature(sid, current_pid, env) for sid in available_sids],
            dtype=np.float32,
        )
        action_idx = self.algo.take_action(features, force_greedy=not self.train_mode)

        if self.train_mode:
            sid = available_sids[action_idx]
            reward = 0.0
            if self.use_shaping_reward:
                reward = self._action_shaping_reward(sid, current_pid, env) * self.shaped_reward_scale
            self.algo.transition['states'].append(features)
            self.algo.transition['actions'].append(action_idx)
            self.algo.transition['rewards'].append(reward)
            self._add_reward_stat('shape', reward)
            self._add_reward_stat('reposition_count', 1)

        return available_sids[action_idx]

    def after_every_episode(self, env: Environment):
        if not self.train_mode:
            return

        self._assign_previous_cost_reward(env)
        if self.use_cost_reward and self.algo.transition['rewards']:
            overtime_cost = env.station_overtime_cost + env.project_overtime_cost
            overtime_penalty = overtime_cost * self.cost_reward_scale / len(self.algo.transition['rewards'])
            for i in range(len(self.algo.transition['rewards'])):
                self.algo.transition['rewards'][i] -= overtime_penalty
            self._add_reward_stat('overtime', -overtime_penalty * len(self.algo.transition['rewards']))

        total_reward = sum(self.algo.transition['rewards'])
        print_with_time(f'ScratchRL Reposition cum reward = {total_reward}')
        base_reward = (
            self.reward_stats['shape']
            + self.reward_stats['cost_delta']
            + self.reward_stats['overtime']
        )
        print_with_time(
            "Scratch reward stats: "
            f"base={base_reward:.4f}, shape={self.reward_stats['shape']:.4f}, "
            f"cost_delta={self.reward_stats['cost_delta']:.4f}, "
            f"overtime={self.reward_stats['overtime']:.4f}, "
            f"perception_raw={self.reward_stats['perception_raw']:.4f}, "
            f"perception_scaled={self.reward_stats['perception_scaled']:.4f}, "
            f"repositions={int(self.reward_stats['reposition_count'])}, "
            f"perception_events={int(self.reward_stats['perception_event_count'])}"
        )
        self.algo.update()
        if (env.i_episode + 1) % SAVE_MODEL_FREQUENCY == 0:
            self.algo.save(self.model_save_dir, env.i_episode)
            print_with_time("ScratchRL Reposition model saved")


class ScratchDispatchAwareRLRepositionAgent(ScratchRLRepositionAgent):
    """从零训练的 Dispatch-aware Reposition RL。"""

    model_save_dir = MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_SAVE_DIR
    best_model_epoch = MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_BEST_EPOCH
    perception_reward_scale = 1.0
    use_cost_reward = True
    use_shaping_reward = True
    use_perception_reward = True

    def apply_perception_rewards(self, perception_layer: PerceptionLayer, env: Environment):
        if not self.train_mode:
            return
        if not self.use_perception_reward:
            return
        if not perception_layer.dispatch_records:
            return

        last_dispatch = perception_layer.dispatch_records[-1]
        positive_rewards, negative_rewards = perception_layer.process_dispatch_result(env, last_dispatch)

        for repo_idx, reward in positive_rewards:
            if repo_idx < len(self.algo.transition['rewards']):
                scaled_reward = self.perception_reward_scale * reward
                self.algo.transition['rewards'][repo_idx] += scaled_reward
                self._add_reward_stat('perception_raw', reward)
                self._add_reward_stat('perception_scaled', scaled_reward)
                self._add_reward_stat('perception_event_count', 1)

        for repo_idx, reward in negative_rewards:
            if repo_idx < len(self.algo.transition['rewards']):
                scaled_reward = self.perception_reward_scale * reward
                self.algo.transition['rewards'][repo_idx] += scaled_reward
                self._add_reward_stat('perception_raw', reward)
                self._add_reward_stat('perception_scaled', scaled_reward)
                self._add_reward_stat('perception_event_count', 1)


class ScratchDispatchAwareNoCostRLRepositionAgent(ScratchDispatchAwareRLRepositionAgent):
    """ScratchDispatchAwareRL消融：去掉成本增量和加班成本reward，保留shaping和感知奖励。"""

    model_save_dir = MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_COST_SAVE_DIR
    best_model_epoch = MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_COST_BEST_EPOCH
    use_cost_reward = False
    use_shaping_reward = True


class ScratchDispatchAwareNoShapingRLRepositionAgent(ScratchDispatchAwareRLRepositionAgent):
    """ScratchDispatchAwareRL消融：去掉action shaping reward，保留成本和感知奖励。"""

    model_save_dir = MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_SHAPING_SAVE_DIR
    best_model_epoch = MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_SHAPING_BEST_EPOCH
    use_cost_reward = True
    use_shaping_reward = False


class ScratchCombinedRLRepositionAgent(ScratchDispatchAwareRLRepositionAgent):
    """独立存储的Scratch组合奖励RL，用校准系数融合成本/shape和感知奖励。"""

    model_save_dir = MODEL_REPOSITION_SCRATCH_COMBINED_SAVE_DIR
    best_model_epoch = MODEL_REPOSITION_SCRATCH_COMBINED_BEST_EPOCH
    perception_reward_scale = SCRATCH_COMBINED_PERCEPTION_REWARD_SCALE


class ScratchCostOnlyRLRepositionAgent(ScratchRLRepositionAgent):
    """Scratch reward消融：只保留成本增量和加班成本reward。"""

    model_save_dir = MODEL_REPOSITION_SCRATCH_COST_ONLY_SAVE_DIR
    best_model_epoch = MODEL_REPOSITION_SCRATCH_COST_ONLY_BEST_EPOCH
    use_cost_reward = True
    use_shaping_reward = False
    use_perception_reward = False


class ScratchPerceptionOnlyRLRepositionAgent(ScratchDispatchAwareRLRepositionAgent):
    """Scratch reward消融：只保留感知层reward。"""

    model_save_dir = MODEL_REPOSITION_SCRATCH_PERCEPTION_ONLY_SAVE_DIR
    best_model_epoch = MODEL_REPOSITION_SCRATCH_PERCEPTION_ONLY_BEST_EPOCH
    use_cost_reward = False
    use_shaping_reward = False
    use_perception_reward = True
