from abc import ABC, abstractmethod
from simulator import Environment
from component import *


class BaseRepositionAgent(ABC):
    """重定位Agent基类，定义统一的训练/测试生命周期接口"""

    def __init__(self, train_mode: bool = False):
        self.train_mode = train_mode

    @abstractmethod
    def select_reposition_station(self, current_pid, truck: Truck, available_sids, env: Environment):
        """选择一个厂站进行重新定位"""
        pass

    def model_initialization(self, load_episode=-1) -> int:
        """模型初始化，返回起始episode编号"""
        return 0

    def before_every_episode(self) -> None:
        """每个episode开始前的准备工作"""
        pass

    def after_every_episode(self, env: Environment) -> None:
        """每个episode结束后的收尾工作"""
        pass

    def after_every_step(self, env: Environment) -> None:
        """每个步骤结束后的收尾工作"""
        pass
