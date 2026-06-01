from parameter import *

from utils import *
import bisect
import math
import numpy as np
import datetime
import pandas as pd
from typing import List
from bisect import bisect_left, bisect_right
from toolkit.geocal import coordinate_to_xy


from enum import Enum

class TruckState(Enum):
    Free = 0
    Load = 1
    Transport = 2
    Cast = 3
    Return = 4

class Truck:
    
    id_count = 0
    
    def __init__(self, oid, from_sid, to_pid, ret_sid) -> None:
        
        self.id = Truck.id_count
        
        Truck.id_count += 1
        self.oid = oid
        self.from_sid = from_sid
        self.to_pid = to_pid
        self.ret_sid = ret_sid
        
        self.state = TruckState.Free
        self.left_time = 0
        self.is_first_dispatch = False

class TruckIterator:
    def __init__(self) -> None:
        
        self.sorted_list:List[Truck] = []
        
    def __bool__(self):
        return bool(self.sorted_list)  # 如果 items 列表非空，则返回 True
        
    def insert_in_order(self,new_truck:Truck):
        import bisect
        index = bisect.bisect_left(self.sorted_list, new_truck.left_time, key=lambda t:t.left_time)
        self.sorted_list.insert(index, new_truck)
        
    def pass_time(self,dt):
        for truck in self.sorted_list:
            truck.left_time-=dt
            
    def next_event_lt(self):
        if self.sorted_list:  # 确保列表非空
            return self.sorted_list[0].left_time
        return float('inf')  # 如果列表为空，返回无穷大或其他适当值
    
    def return_next_event(self):
        truck = self.sorted_list.pop(0)
        return truck

class ProductionLine:
    def __init__(self,size):
        self.size = size
        self.lines = np.zeros(self.size)
        
    def reset(self):
        self.lines = np.zeros(self.size)
    
    def add_truck_waiting(self):       
        index = np.argmin(self.lines)
        self.lines[index]+=LOAD_CONCRETE_TIME
        return self.lines[index]
    
    def wait_time(self):
        index = np.argmin(self.lines)
        return self.lines[index]+LOAD_CONCRETE_TIME
            
    def step_time(self,time):
        self.lines = np.maximum(self.lines-time,0)

# 某个工地项目
class Project:
    
    # 静态成员
    id_map = {}
    
    
    def __init__(self,pid,project_name,coord):
        self.pid = pid
        self.project_name = project_name
        self.coord = coord # 元组类型(lon,lat)
        self.x,self.y = coordinate_to_xy(*coord)
        self.last_working_time = None  # 工地最后工作时间点

    def __repr__(self):
        return f'pid:{self.pid},name:{self.project_name},coord:{self.coord},xy:({self.x},{self.y})'
        


class Station:
    
    # 静态成员，用以进行id映射，原始的sid长而无序，转换为S0、S1...，并且可以通过Station.id_map进行查询  
    id_map = {}
    
    
    def __init__(self,sid,coord):
        
        
        # 实例成员
        
        self.sid = sid
        # 初始化厂站车辆
        self.init_truck_num = STATION_INIT_TRUCK_NUM
        self.coord = coord
        self.x,self.y = coordinate_to_xy(*coord)
        self.reset()
        
        
    def reset(self):

        self.truck_num = self.init_truck_num
        
        self.product_line = ProductionLine(STATION_PRODUCTION_LINE_SIZE)
        
            
        self.dispatch_record = []
        
        self.return_time_list = []
        # 最近一次服务的工地
        self.recent_serve_pid = 0
        
        self.arrange_dispatch = []
        
        
        self.unsolved_dispatch :List[Dispatch]= []

        self.last_working_time = None

        self.loading_trucks :List[Truck]= []

        
        
    def have_truck(self):
        return self.truck_num>0
    
    def dispatch_truck(self, oid, pid, current_time, is_first_dispatch=False):

        if self.truck_num <= 0:
            return None

        self.truck_num -= 1
        truck = Truck(oid, from_sid=self.sid, to_pid=pid, ret_sid=self.sid)

        truck.state = TruckState.Load
        truck.is_first_dispatch = is_first_dispatch
        wait_time = self.product_line.add_truck_waiting()
        truck.left_time = wait_time
        self.dispatch_record.append(current_time)
        self.last_working_time = current_time

        self.loading_trucks.append(truck)

        if is_first_dispatch:
            self._reorder_loading_queue(truck)

        return truck

    def _reorder_loading_queue(self, new_truck):
        """首车插队：新到的首车与生产线上最早的非首车交换oid/pid"""
        # 找到所有非首车的Loading车辆
        non_first = [t for t in self.loading_trucks if not t.is_first_dispatch]
        if not non_first:
            return

        # 找到left_time最小的非首车（最早出生产线）
        earliest = min(non_first, key=lambda t: t.left_time)

        # 新车的left_time如果更小，说明已经排在最前，不需要交换
        if new_truck.left_time < earliest.left_time:
            return

        # 交换oid和to_pid，以及is_first_dispatch标记
        new_truck.oid, earliest.oid = earliest.oid, new_truck.oid
        new_truck.to_pid, earliest.to_pid = earliest.to_pid, new_truck.to_pid
        new_truck.is_first_dispatch, earliest.is_first_dispatch = earliest.is_first_dispatch, new_truck.is_first_dispatch

    def receive_truck(self, truck: Truck):
        self.truck_num += 1
        self.return_time_list.pop(0)

    def step_time(self,time):
        self.product_line.step_time(time)
        self.return_time_list = [max(t - time, 0) for t in self.return_time_list]
        
    # 给定现在的时间和查询范围（minute）
    def count_dispatch_given_range(self,current_time,time_range):
        
        begin_time = current_time-datetime.timedelta(minutes=time_range)
        
        for index in range(len(self.dispatch_record)-1,-1,-1):
            record_time = self.dispatch_record[index]
            if record_time < begin_time:
                break
        
        else:
            index = -1

        return len(self.dispatch_record)-1-index


    def count_future_dispatch_given_range(self, current_time, time_range):

        # 计算结束时间
        end_time = current_time + datetime.timedelta(minutes=time_range)

        # 使用 bisect 找到开始时间和结束时间的索引
        start_index = bisect_left(self.arrange_dispatch, current_time)
        end_index = bisect_right(self.arrange_dispatch, end_time)

        # 统计范围内的元素数量
        return end_index - start_index


    def add_return_truck(self,return_time):

        bisect.insort(self.return_time_list,return_time)

    def get_next_return_time(self):
        
        if self.return_time_list:
            return self.return_time_list[0]
        else:
            return 240
        
    def __repr__(self):
        return f'sid:{self.sid},x:{self.x},y:{self.y}'




def delete_duplicate_stations(stations_coord_dict:dict[str,tuple]):
    
    close_pairs = []
    station_ids = list(stations_coord_dict.keys())

    station_ids_set = set(station_ids)
    assert len(station_ids) == len(station_ids_set)
    
    for i in range(len(station_ids)):
        for j in range(i + 1, len(station_ids)):
            sid1 = station_ids[i]
            sid2 = station_ids[j]
            
            
            lon1,lat1 = stations_coord_dict[sid1]
            lon2, lat2 = stations_coord_dict[sid2]
            
            dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
            
            
            if dist < 0.015:
                close_pairs.append((sid1, sid2))
                station_ids_set.discard(sid2)  # 删除重复的站点ID
                station_ids_set.discard(sid1)  # 删除重复的站点ID


    clusters :list[set]= []


    for sid1,sid2 in close_pairs:
        
        for cluster in clusters:
            if sid1 in cluster or sid2 in cluster:
                cluster.add(sid1)
                cluster.add(sid2)
                break
        else:
            clusters.append(set([sid1, sid2]))
    
    
    
    
    stations_dict :dict[str,Station] = {}
    new_id = 0
    
            
    # print(f"找到 {len(close_pairs)} 对距离相近的厂站：")
    for cluster in clusters:
        
        center_lat = sum(stations_coord_dict[sid][1] for sid in cluster) / len(cluster)
        center_lon = sum(stations_coord_dict[sid][0] for sid in cluster) / len(cluster)
        
        new_sid = f"S{new_id}"
        new_id+=1
        
        
        stations_dict[new_sid] = Station(new_sid,(center_lon,center_lat))
        
        for sid in cluster:
            Station.id_map[sid] = new_sid
            
            
    for sid in station_ids_set:
        new_sid = f"S{new_id}"
        new_id+=1
        
        stations_dict[new_sid] = Station(new_sid,stations_coord_dict[sid])
        
        Station.id_map[sid] = new_sid
        
        
    return stations_dict
       
    

class Order:
    def __init__(self,oid,pid,quantity,n_need,pouring_type,plan_arrive_time):
        self.oid = oid
        self.pid = pid
        self.quantity = quantity
        self.n_need = n_need
        self.n_already_dispatched = 0
        self.n_already_finished = 0
        # 计划需要到达的时间，早到需要等待，晚到有惩罚
        self.plan_arrive_time :datetime.datetime= plan_arrive_time
        self.finish_time = None
        
        
        self.pouring_type = pouring_type
        
        
        # 上一次浇筑完毕的时间
        self.last_cast_time :datetime.datetime= None
        # 有一个bug，引入上一次到达开始浇筑的时间
        
        self.last_arrive_time :datetime.datetime= None
        
        # 有多少次超时30min
        self.overtime_count = 0
        
        # 收入为每方商砼350元
        self.revenue = self.quantity*CONCRETE_REVENUE_PER_FANG
        
        
        
        
        self.whether_arrive = False
        
        
    def __repr__(self):
        # 创建一个字符串表示，包含所有类成员
        members = [f"{key}: {value}" for key, value in self.__dict__.items()]
        return " ".join(members)
    
    

import copy
class OrderIterator:
    def __init__(self,orders:List[Order]):
        self.orders:List[Order] = orders
        
    def __bool__(self):
        return bool(self.orders)  # 如果 items 列表非空，则返回 True
    
    def next_order_lt(self,ts):
    
        if self.orders:
            order_ts = self.orders[0].plan_arrive_time
            # 按minute返回下个订单的时间
            return ((order_ts-ts).total_seconds())//60
        else:
            return float('inf')
    
    def return_next_order(self):
        return self.orders.pop(0)
    
    
    
class Dispatch:
    def __init__(self, oid, pid, from_sid, dispatch_time):
        self.oid = oid
        self.pid = pid
        self.from_sid = from_sid
        self.dispatch_time: datetime.datetime = dispatch_time
        self.real_dispatch_time: datetime.datetime = None
        self.is_first_dispatch = False


class DispatchIterator:
    def __init__(self,dispatchs):
        self.dispatchs :List[Dispatch]= dispatchs
        self.sort_by_dispatch_time()
        
    def __bool__(self):
        return bool(self.dispatchs)  # 如果 items 列表非空，则返回 True
    
    
    def insert(self,new_dispatch:Dispatch):
        self.dispatchs = [new_dispatch]+self.dispatchs

    def sort_by_dispatch_time(self):
        self.dispatchs.sort(key=lambda d: d.dispatch_time)
    
    
    def next_dispatch_lt(self,ts):
        
        if self.dispatchs:
        
            dispatch_ts = self.dispatchs[0].dispatch_time
            # 按minute返回下个调度的时间
            return ((dispatch_ts-ts).total_seconds())//60
        else:
            return float('inf')
    
    def return_next_dispatch(self):
        
        d = self.dispatchs.pop(0)

        return d
    
    
    
    
    