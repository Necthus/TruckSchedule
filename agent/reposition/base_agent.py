


from abc import ABC,abstractmethod
from simulator import Environment

from component import *


class BaseRepositionAgent(ABC):
	def __init__(self):
		pass

	def select_reposition_station(self,current_pid,truck:Truck,avaliable_sids,env:Environment):
		"""
		选择一个厂站进行重新定位
		:param current_project: 当前项目的相关信息（如位置、需求等）
		:param truck: 当前卡车信息
		:param available_stations: 可供选择的厂站列表
		:param env: 当前环境状态（如交通状况、天气等）
		:return: 选择的厂站ID
		"""
  
		pass
		
  
  
	