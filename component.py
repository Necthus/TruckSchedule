from utils import *
import bisect
import math
import numpy as np
import datetime
import pandas as pd
from typing import List

LOADING_TIME = 15
truck_speed = 0.2 #km/min

from enum import Enum

class TruckState(Enum):
    Free = 0
    Load = 1
    Transport = 2
    Cast = 3
    Return = 4

class Truck:
    
    id_count = 0
    
    def __init__(self,oid,from_sid,to_pid,ret_sid) -> None:
        
        self.id = Truck.id_count
        
        Truck.id_count+=1
        
        self.oid = oid

        self.from_sid = from_sid
        self.to_pid = to_pid
        self.ret_sid = ret_sid
        
        self.state = TruckState.Free
        self.left_time = 0
        
class TruckList:
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
        self.lines[index]+=LOADING_TIME
        return self.lines[index]
    
    def wait_time(self):
        index = np.argmin(self.lines)
        return self.lines[index]+LOADING_TIME
            
    def step_time(self,time):
        self.lines = np.maximum(self.lines-time,0)
        

class Station:
    def __init__(self,sid,truck_num,line_size,coord):
        self.sid = sid
        # 初始化厂站车辆
        
        self.init_truck_num = truck_num
        
        self.truck_num = truck_num
        self.product_line = ProductionLine(line_size)
        self.coord = coord
        # 被调度回来的车，回来+1，调度走-1
        self.return_num = 0 
        self.predict_return_list = []
        self.dispatch_record = []
        
        self.return_time_list = []
        
    def reset(self):
        self.truck_num = self.init_truck_num
        self.product_line.reset()
        self.return_num = 0
        self.predict_return_list = []
        self.dispatch_record = []
        self.return_time_list = []    
        
    def have_truck(self):
        return self.truck_num>0
    
    def dispatch_truck(self,oid,pid,current_time):
        
        if self.truck_num<=0:
            
            return None,-1
        
        else:
            
            truck = Truck(oid,self.sid,pid,self.sid)
            truck.state = TruckState.Load
            wait_time = self.product_line.add_truck_waiting()
            truck.left_time = wait_time
            self.dispatch_record.append(current_time)
            
            # 这代表当前车辆数大于回来的车辆，不是回来的功劳
            if self.truck_num>self.return_num:
                self.truck_num-=1
                return truck,0
            # 这代表当前车辆数等于回来的车辆，是回来的功劳
            else:
                self.truck_num-=1
                self.return_num-=1
                return truck,1
            
    def receive_truck(self):
        self.return_num+=1
        self.truck_num+=1
        # 删除第一个返回时间，作为到达
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
    
    
    
    def add_return_truck(self,return_time):
        
        bisect.insort(self.return_time_list,return_time)
        
        
    def get_next_return_time(self):
        
        if self.return_time_list:
            return self.return_time_list[0]
        else:
            return 240
        
    def __repr__(self):
        return f'sid:{self.sid},coord:{self.coord}'

class Order:
    def __init__(self,oid,pid,quantity,n_need,create_time):
        self.oid = oid
        self.pid = pid
        self.quantity = quantity
        self.n_need = n_need
        self.n_dispatch = 0
        self.n_finish = 0
        self.create_time  = create_time
        self.finish_time = datetime.datetime(2000,1,1)
    def __repr__(self):
        # 创建一个字符串表示，包含所有类成员
        members = [f"{key}: {value}" for key, value in self.__dict__.items()]
        return " ".join(members)
    

        
class OrderIterator:
    def __init__(self,orders):
        self.orders = orders.copy()
        
    def __bool__(self):
        return bool(self.orders)  # 如果 items 列表非空，则返回 True
    
    def next_order_lt(self,ts):
        
        
        if self.orders:
        
            order_ts = self.orders[0]['create_time']
            # 按minute返回下个订单的时间
            return ((order_ts-ts).total_seconds())//60
        else:
            return float('inf')
    
    def return_next_order(self):
        
        raw_order = self.orders.pop(0)
        
        ct = raw_order['create_time']
        oid = raw_order['id']
        pid = raw_order['project_id']
        q = raw_order['order_quantity']
        count = raw_order['ticket_count']
        
        return Order(oid,pid,q,count,ct) 