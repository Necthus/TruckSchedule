


from abc import ABC,abstractmethod
from simulator import Environment
from agent.reposition.base_agent import BaseRepositionAgent
from component import *


class RetraceRepositionAgent(BaseRepositionAgent):
    def __init__(self):
        pass

    def select_reposition_station(self,current_pid,truck:Truck,avaliable_sids,env:Environment):
        from_sid = truck.from_sid
    
        return from_sid
        
  
  
    