from abc import ABC, abstractmethod
from simulator import Environment
from agent.dispatch.base_agent import BaseDispatchAgent
from component import *


class FastestDispatchAgent(BaseDispatchAgent):
    def __init__(self):
        pass

    def select_a_station(self, order: Order, available_sids, env: Environment):

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

