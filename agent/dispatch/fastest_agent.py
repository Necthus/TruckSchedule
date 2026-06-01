from abc import ABC, abstractmethod
from simulator import Environment
from agent.dispatch.base_agent import BaseDispatchAgent
from component import *


class FastestDispatchAgent(BaseDispatchAgent):
    def __init__(self):
        pass

    def get_intent_station(self, order: Order, available_sids, env: Environment):
        """原始意图：理论上最快到达工地的厂站（不考虑实际车辆可用性）"""
        current_pid = order.pid
        l = []
        for sid in available_sids:
            s = env.stations[sid]
            dist = env.interaction[current_pid][sid]
            driving_time = dist / TRUCK_SPEED
            waiting_time = s.product_line.wait_time()
            total_time = driving_time + waiting_time
            l.append((total_time, sid))
        l.sort(key=lambda x: x[0])
        return l[0][1]

    def select_a_station(self, order: Order, available_sids, env: Environment):
        """实际选择：考虑车辆可用性的最快厂站"""
        current_pid = order.pid

        new_available_sids = [sid for sid in available_sids if env.stations[sid].truck_num > 0]

        if new_available_sids == []:
            pass
        else:
            available_sids = new_available_sids

        l = []
        for sid in available_sids:
            s = env.stations[sid]
            dist = env.interaction[current_pid][sid]
            driving_time = dist / TRUCK_SPEED
            waiting_time = s.product_line.wait_time()
            total_time = driving_time + waiting_time
            l.append((total_time, sid))

        l.sort(key=lambda x: x[0])
        select_sid = l[0][1]

        return select_sid
