


from abc import ABC,abstractmethod
from simulator import Environment
from agent.reposition.base_agent import BaseRepositionAgent
from component import *


class UrgentRepositionAgent(BaseRepositionAgent):
	def __init__(self):
		pass

	def select_reposition_station(self,current_pid,truck:Truck,avaliable_sids,env:Environment):
		avaliable_sids.sort(key=lambda s: (env.stations[s].truck_num,env.interaction[current_pid][s]))

		return avaliable_sids[0]
		
  
  
	