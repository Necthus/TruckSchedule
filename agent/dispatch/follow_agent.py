from abc import ABC, abstractmethod
from simulator import Environment
from agent.dispatch.base_agent import BaseDispatchAgent
from component import *


class FollowDispatchAgent(BaseDispatchAgent):
    def __init__(self):
        pass

    def get_intent_station(self, order: Order, available_sids, env: Environment):
        """原始意图：跟随上次服务的厂站"""
        current_pid = order.pid
        if current_pid in env.project_lastest_connect_station:
            return env.project_lastest_connect_station[current_pid]
        return np.random.choice(available_sids)

    def select_a_station(self, order: Order, available_sids, env: Environment):
        current_pid = order.pid

        if current_pid in env.project_lastest_connect_station:
            sid = env.project_lastest_connect_station[current_pid]
        else:
            sid = np.random.choice(available_sids)

        return sid
