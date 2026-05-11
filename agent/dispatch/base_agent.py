


from abc import ABC,abstractmethod
from simulator import Environment

from component import *


class BaseDispatchAgent(ABC):
	def __init__(self):
		pass
	

	def select_a_station_and_a_truck(self,order:Order,available_sids,env:Environment):
     
				
     
     
		"""
		选择一个厂站和一辆车进行派单
		:param order: 当前订单的相关信息（如位置、需求等）
		:param current_pid: 当前项目ID
		:param available_stations: 可供选择的厂站列表
		:param env: 当前环境状态（如交通状况、天气等）
		:return: 选择的厂站ID和卡车ID
		"""
	
		pass
		
  
  
	