from abc import ABC, abstractmethod
from simulator import Environment
from component import *


class BaseDispatchAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_a_station(self, order: Order, available_sids, env: Environment):
        """选择一个厂站进行派单，返回实际选择的sid"""
        pass

    def get_intent_station(self, order: Order, available_sids, env: Environment):
        """获取原始意图厂站（不考虑实际车辆可用性）"""
        return self.select_a_station(order, available_sids, env)
