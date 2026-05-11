


from abc import ABC,abstractmethod
from simulator import Environment
from agent.dispatch.base_agent import BaseDispatchAgent
from component import *


class FastestDispatchAgent(BaseDispatchAgent):
    def __init__(self):
        pass
    

    def select_a_station_and_a_truck(self,order:Order,available_sids,env:Environment):
     
        current_pid = order.pid
  
  
        new_available_sids = [sid for sid in available_sids if env.stations[sid].truck_num > 0]
          
        if new_available_sids == []:
            # 没有可用的厂站，强制开放所有厂站可用
            # return None, None
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
        s = env.stations[select_sid]

        
            
        
        available_trucks = s.truck_list

        truck_sort_list = []
        
        for truck in available_trucks:
            
            dist1 = env.station_dist_matrix[(truck, select_sid)]
            dist2 = env.interaction[current_pid][select_sid]
            
            
            close_more_dist = dist1 -dist2
            
            truck_sort_list.append((close_more_dist, truck))
            
        truck_sort_list.sort(key=lambda x: x[0],reverse=True)    
        
        
        select_sid = l[0][1]
        
        if not s.truck_list:
            select_truck = select_sid
        else:
            select_truck = truck_sort_list[0][1]    
        
        
        return select_sid,select_truck
    
        
  
  
    