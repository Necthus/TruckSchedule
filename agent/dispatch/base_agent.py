from abc import ABC, abstractmethod
from simulator import Environment
from component import *


class BaseDispatchAgent(ABC):
    def __init__(self):
        pass

    def select_a_station(self, order: Order, available_sids, env: Environment):
        """选择一个厂站进行派单"""
        pass

