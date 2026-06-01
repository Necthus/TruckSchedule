from abc import ABC, abstractmethod
from simulator import Environment
from agent.reposition.base_agent import BaseRepositionAgent
from component import *


class UrgentRepositionAgent(BaseRepositionAgent):
    def __init__(self, train_mode=False):
        super().__init__(train_mode)

    def select_reposition_station(self, current_pid, truck: Truck, available_sids, env: Environment):
        available_sids.sort(key=lambda s: (env.stations[s].truck_num, env.interaction[current_pid][s]))
        return available_sids[0]
