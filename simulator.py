
import datetime
from utils import *
from component import *
import numpy as np


OIL_PRICE = 7.2 # 油价 元/升
FUEL_CONSUMPTION = 0.4 # 油耗 升/公里
WAITING_FUEL_CONSUMPTION = 0.333 # 在工地外等待时的油耗 升/分钟

DEMOLITION_COST = 200 # 拆除花费 元/方

UNLOAD_CONCRETE_TIME = 5 # 浇筑混凝土时间 分钟

TRUCK_SPEED = 0.5 # 车辆速度 km/minute 即30km/h

EXTRA_HOUR = 1


DISCONTINUITY_LIMIT_MINUTE = 60
OVER_TIME_LIMIT_MINUTE = 120




class Environment:
    
    
    def __init__(self,truck_per_station =10,line_size = 2,running_day = 1,start_date = datetime.datetime(2024,5,1)) -> None:
        self.shenzhen_range = [113.770,114.334,22.47,22.803]
        
        
        # 先设置为1天
        self.start_date = start_date
        self.end_date = start_date+datetime.timedelta(days=running_day)
        
        
        self.working_range = (6,21)
        self.order_range = (self.working_range[0]+EXTRA_HOUR,self.working_range[1]-EXTRA_HOUR)
        
        self.current_time = self.start_date

    
        self.width, self.height = calculate_range_dimensions(self.shenzhen_range)
        
        self.size_per_grid = 0.5 # km
        self.truck_per_station = truck_per_station
        self.line_size = line_size
        
        self.projects = {}
        self.stations :dict[str,Station]= {}
        self.orders = []
        
        # 以类形式存储的Order
        self.orders_class:List[Order] = []
        
        '''两个字典，以日期为键存储着每天的即时订单和计划订单'''
        self.plan_orders_dict_by_date :dict[datetime.datetime,List[Order]]= {}
        self.instant_orders_dict_by_date :dict[datetime.datetime,List[Order]]= {}
        
        
        # 这个指的是未完全调度的订单，即还需要派出车辆
        self.unscheduled_orders:dict[str,Order] = {} # 是下面dict的子集
        # 这个指的是未完成的订单，还没有完全浇筑完，可能车辆已经全部派出，也可能未完全派出。
        self.unfinished_orders:dict[str,Order] = {} 
        
        self.finished_orders:dict[str,Order] = {}
        
        self.interaction = {}
        self.init_project()
        self.init_station()
        self.init_orders2()
        self.init_interaction2()
        # self.split_orders(know_all_in_advance=True)
        
        self.order_iter = OrderIterator(self.orders)
        
        self.truck_list = TruckList()
        
        self.unfinished_dispatch :List[Dispatch] = []
        
        self.project_lastest_connect_station:dict[str,str] = {}
        
        self.reward_in_a_step = 0
        self.dispatch_out_in_a_step = 0
        self.total_distance = 0
        
        self.log_file = './running.log'
        
        
        
        self.revenue = 0
        self.cost = 0
        self.penalty = {
            'waiting':0,
            'overtime':0,
            'discontinuity':0
        }
        
        self.count = {
            'overtime':0,
            'discontinuity':0
        }
        
        self.fail_dispatch = 0
        self.fail_order = 0
        
        self.must_rep = 0
        
        self.fail_oid_set = set()
        
    def reset(self):
        self.current_time = self.start_date
        
        for sid,s in self.stations.items():
            s.reset()
        
        self.unscheduled_orders = {}
        self.unfinished_orders = {}
        self.finished_orders = {}
        
        self.order_iter = OrderIterator(self.orders)
        self.truck_list = TruckList()
        
        self.reward_in_a_step = 0
        self.dispatch_out_in_a_step = 0
        self.total_distance = 0
        
        
    
    
    
    def reset_statistic_dict(self,d):
        d={key:0 for key in d}
        
        
    def reset2(self):
        
        self.revenue = 0
        self.cost = 0
        self.total_distance = 0
        self.reset_statistic_dict(self.count)
        self.reset_statistic_dict(self.penalty)
        
        for sid,s in self.stations.items():
            s.reset()
        
        self.fail_oid_set = set()
        
        
    def reset_stations(self):
        for sid,s in self.stations.items():
            s.reset()
        
    def change_truck_num(self,truck_num):
        for sid,s in self.stations.items():
            s.truck_num = truck_num
        
    def count_time_per_fang(self):
    
        total_q = 0
        total_dt = 0
        
        for oid,o in self.finished_orders.items():
            q = o.quantity
            
            dt = round((o.finish_time-o.create_time).total_seconds()/60)
            
            total_q+=q
            total_dt+=dt
            
        return total_dt/total_q   
    
    def count_total_money(self):
        return sum(self.penalty.values())+self.cost 
        
    
    def log(self,msg):
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write(f'{self.current_time}: {msg}\n')
            file.flush()
        
    def finish_order_once(self,oid):
        
        # 如果是已经失败的订单，就别管了
        if oid in self.fail_oid_set:
            return 

        order = self.unfinished_orders[oid]
        
        order.n_finish+=1
        # 如果已经运输完成了指定次数
        if order.n_finish == order.n_need:
            # 存档
            self.finished_orders[oid]=order
            order.finish_time = self.current_time
            # 从未完成的里面删除
            del self.unfinished_orders[oid]
            
            # 利润获取
            self.revenue+=order.revenue
            
    def make_order_fail(self,oid):
        
        del self.unfinished_orders[oid]
        if oid in self.unscheduled_orders:
            del self.unscheduled_orders[oid]
            
        self.fail_oid_set.add(oid)
        
               
        
    def split_orders(self,know_all_in_advance=False):
        
        
        plan_orders = [order for order in self.orders if order['pouring_type']!='自卸']
        random_orders = [order for order in self.orders if order['pouring_type']=='自卸']
        
        if know_all_in_advance:
            plan_orders = plan_orders+random_orders
            random_orders = []
            
            
        
        
        for raw_order in plan_orders:
        
            pt = raw_order['deliver_time']
            oid = raw_order['order_id']
            pid = raw_order['project_id']
            q = raw_order['order_quantity']
            count = raw_order['ticket_count']
            
            current_order = Order(oid,pid,q,count,pt) 
            
            self.unfinished_orders[current_order.oid]=current_order
            self.unscheduled_orders[current_order.oid]=current_order
            
        self.orders = random_orders
        
        
    def orders_to_dict_by_date(self,orders:List[Order]):
        
        # orders是按顺序的
        # 首先设定一个初始日期
        current_date = orders[0].plan_arrive_time.date()
        orders_dict_by_date :dict[datetime.datetime,List[Order]]= {}
        
        orders_dict_by_date[current_date]=[]
        
        for order in orders:
            
            # 订单日期
            order_date = order.plan_arrive_time.date()
            
            # 订单日期不在当天
            if order_date != current_date:
                
                # 跳到下一天
                current_date = order_date
                
                # 初始化下一天订单的列表
                orders_dict_by_date[current_date]=[]

            orders_dict_by_date[current_date].append(order)    
            
        return orders_dict_by_date
                
        
    def update_station_feature_from_dispatchs(self,dispatchs:List[Dispatch]):
        
        
        for d in dispatchs:
            
            sid = d.from_sid
            dispatch_time = d.dispatch_time
            self.stations[sid].arrange_dispatch.append(dispatch_time)
        
        
        
    
    def verify_dispatchs(self,dispatchs:List[Dispatch]):
        
        true_need_dict = {}
        dispatch_dict = {}
        
        
        for k,v in self.unscheduled_orders.items():
            
            oid = v.oid
            assert oid == k
            true_need_dict[oid] = v.n_need
        
        for d in dispatchs:
            # 提取该dispatch负责的订单
            oid = d.oid
            if oid not in dispatch_dict:
                dispatch_dict[oid] = 0
            # 订单数量+1
            dispatch_dict[oid] += 1
        
        
        
        
        if true_need_dict!=dispatch_dict:
            print('Dispatchs can not satisfy the orders.')
            return False
        
        # for oid in true_need_dict:
        #     true_need = true_need_dict[oid]
        #     if true_need != dispatch_dict[oid]:
        #         print('Dispatchs can not satisfy the orders.')
        #         print(true_need,dispatch_dict[oid])
        #         return False
        
        print('Dispatchs can satisfy the orders, dispatchs and needs are equal.')
        return True
        
            
    
    # 给定dispatch计划
    # 给定计划外订单的调度方法
    # 给定厂站返程调度方法
    # 运行一天的模拟
    
    def continuous_running(self,dispatchs,instant_orders_dispatcher:callable=None,reposition_method:callable=None,with_random_orders=False):
        
        dispatch_iter = DispatchIterator(dispatchs)
        
        available_stations = 1
        
        if with_random_orders:
            order_iter = self.order_iter
        else:
            order_iter = OrderIterator([])  # 不处理实时订单
        
        
        while True:
            available_stations,features,reward,current_truck  = self.reposition_train_step(dispatch_iter,instant_orders_dispatcher,order_iter)
            
            if available_stations == None:
                # 模拟器运行完毕
                break
            
            # 没有好的reposition方法就返回原厂站
            if reposition_method == None:
                self.take_action(current_truck.ret_sid,current_truck)
            else: 
                # 采用reposition方法
                reposition_station_index = reposition_method(features)
                sid = available_stations[reposition_station_index]
                self.take_action(sid,current_truck)
    
    def running_one_day(self,running_date,dispatchs,instant_orders_dispatch_method:callable=None,reposition_method:callable=None):
        
        '''runningdate指示函数要运行在哪一天'''
        
        '''重置当前时间为今天0点日期'''
        self.current_time = running_date
        
        instant_orders_iter = OrderIterator(self.instant_orders_dict_by_date[running_date.date()])
        
        assert not self.unfinished_orders,'模拟器未开始不应该有未完成的orders'
        assert not self.unscheduled_orders,'模拟器未开始不应该有未调度的orders'
        
        plan_orders_list = self.plan_orders_dict_by_date[running_date.date()]
        
        for order in plan_orders_list:
            
            order_instance = copy.deepcopy(order)
            
            
            oid = order.oid
            self.unfinished_orders[oid]=order_instance
            self.unscheduled_orders[oid]=order_instance
        
        
        '''简单验证一下dispatch的数量是不是对的，后续可更复杂逻辑'''
        self.verify_dispatchs(dispatchs)    
        
        dispatch_iter = DispatchIterator(dispatchs)

        available_stations = 1
        
        while True:
            # 每返回一次代表需要进行一次reposition
            available_stations,features,reward,current_truck  = self.reposition_train_step(dispatch_iter,instant_orders_dispatch_method,instant_orders_iter)
            
            if available_stations == None:
                # 模拟器运行完毕
                break
            
            # 没有好的reposition方法就返回原厂站
            if reposition_method == None:
                self.take_action(current_truck.ret_sid,current_truck)
            else: 
                # 采用reposition方法
                reposition_station_index = reposition_method(features)
                sid = available_stations[reposition_station_index]
                self.take_action(sid,current_truck)
                
        assert not self.unfinished_orders,'模拟器结束后不应该有未完成的orders'
        assert not self.unscheduled_orders,'模拟器结束后不应该有未调度的orders'
    
    
    def reposition_train_step(self,dispatch_iter:DispatchIterator,instant_orders_dispatch_method:callable=None,order_iter:OrderIterator=OrderIterator([])):
        
        '''
        
        Repostion训练的一个step，注意到训练时没有即时订单，所以order_iter很可能为空
        
        TruckList指示车辆事件
        dispatch_iter代表着前一天对今天计划订单指定的Dispatch计划，用于今天
        order_iter代表着即时订单的迭代器，用于表示在系统运行过程中蹦出的那些订单，需要处理，可能为空
        
        '''
        
        pre_total_money = sum(self.penalty.values())+self.cost
        
        
        while self.truck_list or dispatch_iter or order_iter:
            
            event_lt = self.truck_list.next_event_lt()
            dispatch_lt = dispatch_iter.next_dispatch_lt(self.current_time) 
            order_lt = order_iter.next_order_lt(self.current_time)                                       
            
            # 三种事件中最小的时间
            # 0. 车辆事件(状态转换) 1. 发车事件 2. 实时订单事件
            lt_list = [event_lt,dispatch_lt,order_lt]
            index = np.argmin(lt_list)
            lt = lt_list[index]
            
            # 模拟器时间推进
            self.truck_list.pass_time(lt)
            for id,station in self.stations.items():
                station.step_time(lt)
            self.current_time+=datetime.timedelta(minutes=lt)
          

            # 碰到车辆事件了                       
            if index == 0:
                # 车辆状态转换
                
                current_truck = self.truck_list.return_next_event()
    
                assert current_truck.left_time == 0
                
                self.truck_state_change(current_truck)
                
                # 如果车辆不是变成了闲置状态（已经返回到了厂站），就放回到事件队列里面
                if not current_truck.state == TruckState.Free:
                    self.truck_list.insert_in_order(current_truck)
                
                
                # 需要进行返程调度！！！！！！！！！！！！！
                if current_truck.state == TruckState.Return:
                    # 订单完成一次
                    self.finish_order_once(current_truck.oid)
                    
                    
                    # 当前车辆在的工地
                    current_project = current_truck.to_pid
                    # 可用返回的厂站
                    available_stations = list(self.interaction[current_project].keys())
                    # 计算特征
                    # 返回特征，当前时间，之前的奖励，当前需要调度的车辆 
                    # 每一行是一个特征 
                    features = [self.get_action_feature(s,current_project) for s in available_stations]
                    
                    # must_reposition_index = self.must_reposition_stations(features)
                    
                    # if must_reposition_index:
                        
                    #     ret_sid = available_stations[must_reposition_index[0]]
                    #     self.must_rep+=1
                    #     self.take_action(ret_sid,current_truck)  
                        
                    # else:
                        
                    #     now_total_money = sum(self.penalty.values())+self.cost
                    #     reward = -(now_total_money-pre_total_money)
                    #     return available_stations,features,reward,current_truck
                    
                    now_total_money = sum(self.penalty.values())+self.cost
                    reward = -(now_total_money-pre_total_money)
                    return available_stations,features,reward,current_truck
                              
                elif current_truck.state == TruckState.Free:
                    del current_truck
            
            # 碰到发车dispatch了
            elif index == 1:
                # 从迭代器里面取出一个dispatch            
                current_dispatch = dispatch_iter.return_next_dispatch()
                # 首先判断这个订单是不是已经废了，因为超时和浇筑不连续
                serve_oid = current_dispatch.oid
                
                if serve_oid in self.fail_oid_set:
                    # 直接就跳过了
                    continue
                
                # 没问题才继续执行
                # 获取发车的结果
                ret = self.execute_single_dispatch(current_dispatch)
                # 如果发车不成功
                if ret == False:
                    # 记为未完成的dispatch
                    self.fail_dispatch+=1
                    self.unfinished_dispatch.append(current_dispatch)
                    
            # 碰到实时订单了，作为训练step没有，因为是固定订单；训练step作为实际运行的一部分的时候（不训练），会有外部订单。        
            elif index == 2:
                current_order = order_iter.return_next_order()
                
                self.log(current_order.__repr__())
                self.unfinished_orders[current_order.oid]=current_order
                self.unscheduled_orders[current_order.oid]=current_order
                
                # 不存在外部方法，采用最近调度法
                if instant_orders_dispatch_method == None:
                    truck_need = current_order.n_need
                    
                    pid = current_order.pid
                    
                    # 如果有最近记录，选该记录
                    if pid in self.project_lastest_connect_station:
                        chosen_station = self.project_lastest_connect_station[pid]
                    # 否则选最近的厂站
                    else:
                        station_list = [[sid,dist] for sid,dist in self.interaction[pid].items()]
                        chosen_station = min(station_list,key=lambda x:x[1])[0]
                    
                    # 构建新的dispatch
                    for i in range(truck_need):
                        new_dispatch = Dispatch(current_order.oid,current_order.pid,chosen_station,self.current_time,chosen_station)
                        # 放到dispatch队列里面
                        dispatch_iter.dispatchs.insert(0,new_dispatch)
                        
                # 存在外部方法，采用外部方法
                else:
                    # dispatch_list = external_method(current_order)
                    # # 将dispatch_list放到dispatch队列的最前面。注意这里采用了简化操作，假设外部方法的dispatch都是即时的，后面需要改进！！！！！！！！！！！！
                    # dispatch_iter.dispatchs = dispatch_list+dispatch_iter.dispatchs
                    
                    ret = instant_orders_dispatch_method(current_order)
                    # 调度订单
                    # 一次派单全部派完
                    if ret == True:
                        # 从unscheduled里面删去
                        del self.unscheduled_orders[current_order.oid]
                
        
        
        # 已经不存在事件了，模拟器运行完，返回一个状态
        now_total_money = sum(self.penalty.values())+self.cost
        reward = -(now_total_money-pre_total_money)
        return None,None,reward,None      
    
     
          
    
    
    
        
    
    def execute_single_dispatch(self,dispatch:Dispatch):
        
        # 该订单的编号
        oid = dispatch.oid
        
        new_truck,ret = self.stations[dispatch.from_sid].dispatch_truck(oid,dispatch.pid,self.current_time)
 
        # 如果发车不成功
        if new_truck == None:
            # 返回失败
            return False
        # 如果发车成功
        else:
            # 首先把发出的车放到队列里面跑
            new_truck.ret_sid = dispatch.ret_sid
            self.truck_list.insert_in_order(new_truck)
            # 订单已经派出+1
            self.unscheduled_orders[oid].n_dispatch+=1
            assert self.unscheduled_orders[oid].n_dispatch<=self.unscheduled_orders[oid].n_need
            # 记录一下工地的最新联系的厂站
            self.project_lastest_connect_station[dispatch.pid]=dispatch.from_sid
            
            # 如果车辆全部发出
            if self.unscheduled_orders[oid].n_dispatch == self.unscheduled_orders[oid].n_need:
                # 从unscheduled里面删除
                del self.unscheduled_orders[oid]
            
            # 返回成功
            return True
        
    
    
    def execute_unfinished_dispatch(self):
        
        still_unfinished_dispatch = []
        # 遍历未完成的dispatch
        for d in self.unfinished_dispatch:
            # 执行一个dispatch
            ret = self.execute_single_dispatch(d)
            # 仍然未完成
            if ret == False:
                still_unfinished_dispatch.append(d)
        
        # 重新赋值未完成的dispatch        
        self.unfinished_dispatch = still_unfinished_dispatch
            
    
    
    def running(self):
        
        self.log('-----------------------------------------------------')
        self.log('System Begins')
        
        last_time = None
        last_state = None
        last_action = None
        
        sample_sequence = []
        
        while(1):
        
            stations_features,current_time,regu_reward,current_truck  = self.step()
            
            if last_time:
                
                sample_sequence.append((last_time,last_state,last_action,regu_reward))
            
            
            if stations_features == None:
                break
                
            last_state = stations_features
            last_time = current_time
                
            action = np.random.choice(list(stations_features.keys()))
            self.take_action(action,current_truck)
            
            last_action = action
            
        return sample_sequence
        
        
    
    def must_reposition_stations(self,stations_features):
        
        index_list = []
        
        for index,feature in enumerate(stations_features):
            # 这个厂站没车了
            if feature[1]==0 and feature[3]>200:
                index_list.append(index)
                
        return index_list
        
        
    
    
    def take_action(self,return_sid,truck:Truck):
        
        # 设定搅拌车的返回sid
        truck.ret_sid = return_sid
        # 设定返回时间
        truck.left_time = round(self.interaction[truck.to_pid][return_sid]/TRUCK_SPEED)
        # 通知厂站有一个车要回来了
        self.stations[return_sid].add_return_truck(truck.left_time)
        
        
        
    def step(self):
        
        self.reward_in_a_step = 0
        
        self.dispatch_out_in_a_step = 0
        
        while self.order_iter or self.truck_list:
            
            order_lt = self.order_iter.next_order_lt(self.current_time)
            event_lt = self.truck_list.next_event_lt()
            
            
            lt = min(order_lt,event_lt)
            
            self.truck_list.pass_time(lt)
            for id,station in self.stations.items():
                station.step_time(lt)
            self.current_time+=datetime.timedelta(minutes=lt)
            
            # 碰到订单了
            if order_lt<event_lt:
                
                current_order = self.order_iter.return_next_order()
                
                self.log(current_order.__repr__())
                self.unfinished_orders[current_order.oid]=current_order
                self.unscheduled_orders[current_order.oid]=current_order
                # 调度订单
                ret = self.schedule_order(current_order)
                # 一次派单全部派完
                if ret == True:
                    # 从unscheduled里面删去
                    del self.unscheduled_orders[current_order.oid]
                                   
            else:
                # 车辆状态转换
                current_truck = self.truck_list.return_next_event()
    
                assert current_truck.left_time == 0
                
                self.truck_state_change(current_truck)
                
                if not current_truck.state == TruckState.Free:
                    self.truck_list.insert_in_order(current_truck)
                
                
                # 需要进行返程调度！！！！！！！！！！！！！
                if current_truck.state == TruckState.Return:
                    # 订单完成一次
                    self.finish_order_once(current_truck.oid)
                    # 当前车辆在的工地
                    current_project = current_truck.to_pid
                    # 可用返回的厂站
                    available_stations = sorted(self.interaction[current_project],key = lambda x:self.interaction[current_project][x])
                    # 最近的厂站
                    nearest_sid = available_stations[0]
                    
                    reduced_stations = available_stations[0:4]
                    
                    stations_features = {}
                    # 计算特征
                    for s in reduced_stations:
                        
                        stations_features[s]=self.get_action_feature(s,current_project)
                    # 返回特征，当前时间，之前的奖励，当前需要调度的车辆   
                    
                    
                    if self.dispatch_out_in_a_step != 0:
                        regu_reward = self.reward_in_a_step/self.dispatch_out_in_a_step
                    else:
                        regu_reward = 0 
                    return stations_features,self.current_time,regu_reward,current_truck
                    
                    
                elif current_truck.state == TruckState.Free:
                    
                    self.stations[current_truck.ret_sid].truck_num+=1
                    self.stations[current_truck.ret_sid].return_num+=1
                    
                    del current_truck
                    
        if self.dispatch_out_in_a_step != 0:
            regu_reward = self.reward_in_a_step/self.dispatch_out_in_a_step
        else:
            regu_reward = 0           
        return None,self.current_time,regu_reward,None  
                    
    def get_action_feature(self,sid,pid):
        
        s = self.stations[sid]
        
        
        dispatch_count = s.count_dispatch_given_range(self.current_time,60)
        truck_num = s.truck_num
        dist = self.interaction[pid][sid]
        # 特征：过去1小时发车数量，当前车数量，距离工地距离，
        future_dispatch_feature = []
        
        for trange in [30,60,120]:
            future_dispatch_feature.append(s.count_future_dispatch_given_range(self.current_time,trange))
            
        
        return [dispatch_count,truck_num,dist,s.get_next_return_time()]+future_dispatch_feature
                        
            
    def schedule_order_no_reward(self,order:Order):
        
        station_list = [[sid,dist] for sid,dist in self.interaction[order.pid].items()]
        
        sid_go_time_dict = {}
        sid_total_time_dict = {}
        can_dispatch_set = set()
          
        for sid,dist in station_list:
            wt = self.stations[sid].product_line.wait_time()
            gt = round(dist/TRUCK_SPEED)
            
            sid_go_time_dict[sid]=gt
            sid_total_time_dict[sid]=wt+gt
            if self.stations[sid].have_truck():
                can_dispatch_set.add(sid)
        
        
        need = order.n_need-order.n_dispatch
        # 本轮次刚dispatch车的sid
        lsid = -1
        
        for i in range(need):
            
            # 更新本轮次刚dispatch车的staion的状态
            if lsid!=-1:
                wt = self.stations[lsid].product_line.wait_time()
                gt = sid_go_time_dict[lsid]
                sid_total_time_dict[lsid]=wt+gt
                if not self.stations[lsid].have_truck():
                    can_dispatch_set.remove(lsid)
                    
            
            
            optimal_sid = min(sid_total_time_dict, key=lambda k: sid_total_time_dict[k])
            
            actual_sid = min(can_dispatch_set,key=lambda k:sid_total_time_dict[k],default=None)
            
           
            
            # 派单不成功
            if not actual_sid:
                extra_time = sid_total_time_dict[optimal_sid]
                self.reward_in_a_step -=self.stations[optimal_sid].get_next_return_time()
                return False
            # 派单成功
            else:
                # 如果是最优派单
                if actual_sid == optimal_sid:
                    new_truck,r = self.stations[actual_sid].dispatch_truck(order.oid,order.pid,self.current_time)
                    
                    # 并且派单所用到的车是return回来的车
                    if r == 1:
                    
                        copy_set = can_dispatch_set.copy()
                        copy_set.remove(actual_sid)
                        second_actual_sid = min(copy_set,key=lambda k:sid_total_time_dict[k],default=None)
                        
                        if not second_actual_sid:
                            save_time = self.stations[optimal_sid].get_next_return_time()
                        else:
                            save_time = sid_total_time_dict[second_actual_sid]-sid_total_time_dict[optimal_sid]
                            
                        self.reward_in_a_step +=save_time
                
                
                # 如果不是最优派单，即实际调度的车要花费比最优更多时间        
                else:
                    new_truck,r = self.stations[actual_sid].dispatch_truck(order.oid,order.pid,self.current_time)
                
                    extra_time = sid_total_time_dict[actual_sid]-sid_total_time_dict[optimal_sid]
                
                    self.reward_in_a_step -=extra_time
                 
                 
                self.log(f'Send a new truck {new_truck.id} from S {new_truck.from_sid} to P {new_truck.to_pid} for Order {new_truck.oid}') 
                # 加入事件队列 
                self.truck_list.insert_in_order(new_truck)
                # 订单已经派出+1
                order.n_dispatch+=1
                
                self.dispatch_out_in_a_step +=1
                
                lsid = actual_sid
                
        return True        
            
    # 订单调度，可能要派遣多个车辆
    def schedule_order(self,order:Order):
        
        
        # 计算厂站到工地的距离，列表
        station_list = [[sid,dist] for sid,dist in self.interaction[order.pid].items()]
        
        # 计算每个厂站的去程时间和总时间
        sid_go_time_dict = {}
        sid_total_time_dict = {}
        can_dispatch_set = set()
        
        # 遍历厂站列表，计算每个厂站的去程时间和总时间
        for sid,dist in station_list:
            wt = self.stations[sid].product_line.wait_time()
            gt = round(dist/TRUCK_SPEED)
            
            sid_go_time_dict[sid]=gt
            sid_total_time_dict[sid]=wt+gt
            # 如果该厂站有车可以调度
            if self.stations[sid].have_truck():
                # 将该厂站加入可以调度的集合
                can_dispatch_set.add(sid)
        
        
        need = order.n_need-order.n_dispatch
        # 本轮次刚dispatch车的sid
        last_dispatch_sid = -1
        
        for i in range(need):
            
            # 更新本轮次刚dispatch车的staion的状态
            if last_dispatch_sid!=-1:
                # 更新该厂站的等待时间和总时间
                # 这里的wt是厂站的等待时间，gt是去程时间
                wt = self.stations[last_dispatch_sid].product_line.wait_time()
                gt = sid_go_time_dict[last_dispatch_sid]
                sid_total_time_dict[last_dispatch_sid]=wt+gt
                # 如果该厂站没有车了，就从可以调度的集合中删除
                if not self.stations[last_dispatch_sid].have_truck():
                    can_dispatch_set.remove(last_dispatch_sid)
                    
                    
            optimal_sid = min(sid_total_time_dict, key=lambda k: sid_total_time_dict[k])
            actual_sid = min(can_dispatch_set,key=lambda k:sid_total_time_dict[k],default=None)
            
            
            # 派单不成功
            if not actual_sid:
                extra_time = sid_total_time_dict[optimal_sid]
                self.reward_in_a_step -=self.stations[optimal_sid].get_next_return_time()
                return False
            # 派单成功
            else:
                # 如果是最优派单
                if actual_sid == optimal_sid:
                    new_truck,r = self.stations[actual_sid].dispatch_truck(order.oid,order.pid,self.current_time)
                    
                    # 并且派单所用到的车是return回来的车
                    if r == 1:
                    
                        copy_set = can_dispatch_set.copy()
                        copy_set.remove(actual_sid)
                        second_actual_sid = min(copy_set,key=lambda k:sid_total_time_dict[k],default=None)
                        
                        if not second_actual_sid:
                            save_time = self.stations[optimal_sid].get_next_return_time()
                        else:
                            save_time = sid_total_time_dict[second_actual_sid]-sid_total_time_dict[optimal_sid]
                            
                        self.reward_in_a_step +=save_time
                
                
                # 如果不是最优派单，即实际调度的车要花费比最优更多时间        
                else:
                    new_truck,r = self.stations[actual_sid].dispatch_truck(order.oid,order.pid,self.current_time)
                
                    extra_time = sid_total_time_dict[actual_sid]-sid_total_time_dict[optimal_sid]
                
                    self.reward_in_a_step -=extra_time
                 
                 
                self.log(f'Send a new truck {new_truck.id} from S {new_truck.from_sid} to P {new_truck.to_pid} for Order {new_truck.oid}') 
                # 加入事件队列 
                self.truck_list.insert_in_order(new_truck)
                # 订单已经派出+1
                order.n_dispatch+=1
                
                self.dispatch_out_in_a_step +=1
                
                last_dispatch_sid = actual_sid
                
        return True
    
    # 订单调度，可能要派遣多个车辆
    def schedule_order_new(self,order:Order):
        
        
        # 计算厂站到工地的距离，列表
        station_list = [[sid,dist] for sid,dist in self.interaction[order.pid].items()]
        
        # 计算每个厂站的去程时间和总时间
        sid_go_time_dict = {}
        sid_total_time_dict = {}
        can_dispatch_set = set()
        
        # 遍历厂站列表，计算每个厂站的去程时间和总时间
        for sid,dist in station_list:
            wt = self.stations[sid].product_line.wait_time()
            gt = round(dist/TRUCK_SPEED)
            
            sid_go_time_dict[sid]=gt
            sid_total_time_dict[sid]=wt+gt
            # 如果该厂站有车可以调度
            if self.stations[sid].have_truck():
                # 将该厂站加入可以调度的集合
                can_dispatch_set.add(sid)
        
        
        need = order.n_need-order.n_dispatch
        # 本轮次刚dispatch车的sid
        last_dispatch_sid = -1
        
        for i in range(need):
            
            # 更新本轮次刚dispatch车的staion的状态
            if last_dispatch_sid!=-1:
                # 更新该厂站的等待时间和总时间
                # 这里的wt是厂站的等待时间，gt是去程时间
                wt = self.stations[last_dispatch_sid].product_line.wait_time()
                gt = sid_go_time_dict[last_dispatch_sid]
                sid_total_time_dict[last_dispatch_sid]=wt+gt
                # 如果该厂站没有车了，就从可以调度的集合中删除
                if not self.stations[last_dispatch_sid].have_truck():
                    can_dispatch_set.remove(last_dispatch_sid)
                    
                    
            optimal_sid = min(sid_total_time_dict, key=lambda k: sid_total_time_dict[k])
            actual_sid = min(can_dispatch_set,key=lambda k:sid_total_time_dict[k],default=None)
            
            
            # 派单不成功
            if not actual_sid:
                extra_time = sid_total_time_dict[optimal_sid]
                self.reward_in_a_step -=self.stations[optimal_sid].get_next_return_time()
                return False
            # 派单成功
            else:
                # 如果是最优派单
                if actual_sid == optimal_sid:
                    new_truck,r = self.stations[actual_sid].dispatch_truck(order.oid,order.pid,self.current_time)
                    
                    # 并且派单所用到的车是return回来的车
                    if r == 1:
                    
                        copy_set = can_dispatch_set.copy()
                        copy_set.remove(actual_sid)
                        second_actual_sid = min(copy_set,key=lambda k:sid_total_time_dict[k],default=None)
                        
                        if not second_actual_sid:
                            save_time = self.stations[optimal_sid].get_next_return_time()
                        else:
                            save_time = sid_total_time_dict[second_actual_sid]-sid_total_time_dict[optimal_sid]
                            
                        self.reward_in_a_step +=save_time
                
                
                # 如果不是最优派单，即实际调度的车要花费比最优更多时间        
                else:
                    new_truck,r = self.stations[actual_sid].dispatch_truck(order.oid,order.pid,self.current_time)
                
                    extra_time = sid_total_time_dict[actual_sid]-sid_total_time_dict[optimal_sid]
                
                    self.reward_in_a_step -=extra_time
                 
                 
                self.log(f'Send a new truck {new_truck.id} from S {new_truck.from_sid} to P {new_truck.to_pid} for Order {new_truck.oid}') 
                # 加入事件队列 
                self.truck_list.insert_in_order(new_truck)
                # 订单已经派出+1
                order.n_dispatch+=1
                
                self.dispatch_out_in_a_step +=1
                
                last_dispatch_sid = actual_sid
                
        return True
        
    def truck_state_change(self,truck:Truck):
        # 1 去厂站装水泥 2 排队等待（不一定）+装水泥（一定时间） 3 去工地卸水泥 + 卸水泥（不需要排队） 4 返程
        # 车辆从闲置状态开始/不可能
        if truck.state == TruckState.Free:
            
            print('Error',self.current_time)
            raise ValueError('Truck is free, but not in the station.')
            
        # 车辆结束装载状态
        elif truck.state == TruckState.Load:
            
            truck.state = TruckState.Transport
            dist = self.interaction[truck.to_pid][truck.from_sid]    
            truck.left_time = round(dist/TRUCK_SPEED)    
            
        
        # 车辆结束运输阶段，已经到达了工地
        elif truck.state == TruckState.Transport:
            # 车辆服务的订单
            oid = truck.oid
            # 看看距离
            distance = self.interaction[truck.to_pid][truck.from_sid]
            # 计算油钱
            self.cost += distance*FUEL_CONSUMPTION*OIL_PRICE
            
            
            # 先查看一下订单是不是已经失败了，这时候就可以滚蛋了
            
            if oid in self.fail_oid_set:
                truck.state = TruckState.Return
                return 
            
            
            # 变为浇筑状态
            truck.state = TruckState.Cast
            
            current_order = self.unfinished_orders[oid]
            
            # 如果还一次都没有浇筑，说明是第一个到达的车
            if current_order.n_finish == 0:
                plan_time = current_order.plan_arrive_time
                current_time = self.current_time
                # 如果提早到了，计算等待的惩罚油钱
                if current_time<plan_time:
                    dt = plan_time-current_time
                    dt_in_minute = round(dt.total_seconds()/60)
                    self.penalty['waiting']+=dt_in_minute*WAITING_FUEL_CONSUMPTION*OIL_PRICE
                    # 需要额外等待
                    truck.left_time = dt_in_minute
                    
                    
                # 如果没有提前到，看看迟到多少
                else:
                    dt = current_time-plan_time
                    dt_in_minute = dt.total_seconds()/60
                    # # 如果迟到超过2小时，进行比例罚款
                    # if dt_in_minute > 120:
                    #     self.penalty['overtime']+=0.01*current_order.revenue
                    # # 如果迟到超过6小时，进行固定巨额罚款
                    # if dt_in_minute > 360:
                    #     self.penalty['overtime']+=20000

                    # if dt_in_minute > 24*60:
                    #     current_order.revenue = 0
                    #     self.fail_order+=1
                    
                    # 如果迟到超过2小时，直接失败吧
                    if dt_in_minute > 120:
                        self.count['overtime']+=1
                        self.make_order_fail(current_order.oid)
                        truck.state = TruckState.Return
                        

                    
            # 如果已经浇筑过了，说明不是第一个到达的
            else:
                # 上一辆车还没浇筑完，不存在连续浇筑惩罚
                if current_order.last_cast_time == None:
                    pass
                else:
                    dt = self.current_time-current_order.last_cast_time
                    dt_in_minute = dt.total_seconds()/60
                    # if dt_in_minute > 30:
                    #     current_order.overtime_count+=1
                    # if dt_in_minute > 60:
                    #     self.penalty['discontinuity'] += 10*current_order.n_finish*DEMOLITION_COST
                    #     self.fail_order+=1
                        
                    # if current_order.overtime_count>3:
                    #     self.penalty['discontinuity'] += 5000*current_order.overtime_count
                    
                    if dt_in_minute > DISCONTINUITY_LIMIT_MINUTE:
                        self.count['discontinuity']+=1
                        self.make_order_fail(current_order.oid)
                        truck.state = TruckState.Return
                    
            
            truck.left_time += UNLOAD_CONCRETE_TIME
            
            # 到达时将记录走过了多少距离
            self.total_distance+=distance
            
        # 车辆浇筑完了，要返回了 
        elif truck.state == TruckState.Cast:
            # 更新最后一次的浇筑时间
            current_order = self.unfinished_orders[truck.oid]
            current_order.last_cast_time = self.current_time
            
            truck.state = TruckState.Return
               
        # 返回结束了 
        elif truck.state == TruckState.Return:
            
            # 车辆回到厂站，变为可用状态
            truck.state = TruckState.Free
            truck.left_time = 0
            
            sid = truck.ret_sid
            s = self.stations[sid]
            s.receive_truck()
            
            
            # 回到厂站时将记录走过了多少距离
            distance = self.interaction[truck.to_pid][truck.ret_sid]
            # 计算油钱
            self.cost += distance*FUEL_CONSUMPTION*OIL_PRICE
            
            self.total_distance+=distance
            
            # 回到厂站后，由于厂站有车，所以执行剩余的dispatch
            self.execute_unfinished_dispatch()                        
                
            
        self.log(f'Truck {truck.id} changes to State {truck.state}, left time is {truck.left_time}')    
        
         
        
    def lat_lng_to_grid(self,lat,lng):
        km_per_lat = 111
        km_per_lng = 111 * math.cos(math.radians((22.47+22.803) / 2))

        # 获取网格的原点
        origin_lng = self.shenzhen_range[0]  # 左下角的经度
        origin_lat = self.shenzhen_range[2]  # 左下角的纬度

        # 计算网格索引
        x = int((lng - origin_lng)*km_per_lng / self.size_per_grid)
        y = int((lat - origin_lat)*km_per_lat / self.size_per_grid)

    # 返回网格索引
        return x, y    
        
    def init_station(self,filename = './data/shenzhen_stations.csv'):
        
        stations = pd.read_csv(filename,dtype={'id': str})
        
        for index, row in stations.iterrows():
            stationid = row['id']
            x,y = self.lat_lng_to_grid(row['lat'],row['lng'])
            self.stations[stationid]=Station(stationid,self.truck_per_station,self.line_size,(x,y))
        
        
    def init_project(self,filename = './data/shenzhen_projects.csv'):
        
        projects = pd.read_csv(filename,dtype={'id': str})
        
        for index, row in projects.iterrows():
            id = row['id']
            x,y = self.lat_lng_to_grid(row['lat'],row['lng'])
            self.projects[id]=(x,y)
            
    def init_orders(self,filename = './data/shenzhen_orders_0324.csv'):
        fields_to_keep = ['project_id','create_time','deliver_time','order_quantity','ticket_count','pouring_type','order_id']
        type_conversions = {'project_id': str,'order_quantity': float,'ticket_count':int,'order_id':str}
        df = pd.read_csv(filename, usecols=fields_to_keep, dtype=type_conversions,parse_dates=['create_time','deliver_time'])
        
        # 将 Timestamp 转换为 datetime
        # 将 deliver_time 列转换为 datetime 类型
        df['deliver_time'] = pd.to_datetime(df['deliver_time'])
        df['create_time'] = pd.to_datetime(df['create_time'])
        
        # 不妨假设deliver_time是真实的送货时间，以此进行筛选 
        filtered_df = df[(df['deliver_time'] >= self.start_date) & (df['deliver_time'] <= self.end_date)]
        
        self.orders = filtered_df.to_dict(orient='records')
        for index, order in enumerate(self.orders, start=1):
            order['id'] = index
            
            

            
            
    def init_orders2(self,filename = './data/shenzhen_orders_0324.csv'):
        fields_to_keep = ['project_id','create_time','deliver_time','order_quantity','ticket_count','pouring_type','order_id']
        type_conversions = {'project_id': str,'order_quantity': float,'ticket_count':int,'order_id':str}
        df = pd.read_csv(filename, usecols=fields_to_keep, dtype=type_conversions,parse_dates=['create_time','deliver_time'])
        
        # 将 Timestamp 转换为 datetime
        # 将 deliver_time 列转换为 datetime 类型
        df['deliver_time'] = pd.to_datetime(df['deliver_time'])
        df['create_time'] = pd.to_datetime(df['create_time'])
        
        # 不妨假设deliver_time是真实的送货时间，以此进行筛选 
        filtered_df = df[(df['deliver_time'] >= self.start_date) & (df['deliver_time'] <= self.end_date)]
        
        # 并筛选指定时间范围内的订单
        filtered_df = filtered_df[(filtered_df['deliver_time'].dt.hour >= self.order_range[0]) & (df['deliver_time'].dt.hour < self.order_range[1])]
        
        
        self.orders = filtered_df.to_dict(orient='records')
        for index, order in enumerate(self.orders, start=1):
            order['id'] = index
        
        # 分为自卸和其他，作为计划订单和即时订单
        plan_orders = [order for order in self.orders if order['pouring_type']!='自卸']
        random_orders = [order for order in self.orders if order['pouring_type']=='自卸']
        
        # 从字段表转换为类的列表
        def return_orders_class(raw_orders):
            orders_class = []
            for raw_order in raw_orders:
            
                pt = raw_order['deliver_time']
                oid = raw_order['order_id']
                pid = raw_order['project_id']
                q = raw_order['order_quantity']
                count = raw_order['ticket_count']
                
                current_order = Order(oid,pid,q,count,pt) 
                
                orders_class.append(current_order)
                
            return orders_class
        
        plan_orders_class = return_orders_class(plan_orders)
        random_orders_class = return_orders_class(random_orders)
        
        
        self.plan_orders_dict_by_date = self.orders_to_dict_by_date(plan_orders_class)
        self.instant_orders_dict_by_date = self.orders_to_dict_by_date(random_orders_class)
        
        
    def return_next_day_plan_orders(self,today_date:datetime.datetime,day_0 = False):
        
        if day_0:
            
            tomorrow_date = self.start_date
        else:
            tomorrow_date = today_date + datetime.timedelta(days=1)
        tomorrow_plan_orders = self.plan_orders_dict_by_date[tomorrow_date.date()]
        return tomorrow_plan_orders
        
        
    def distance_between_grids(self,x1,y1,x2,y2):
        return (abs(x1-x2)+abs(y1-y2))*self.size_per_grid    
    
    
    
    def init_interaction(self):
        import csv
        project_station_interaction = {}
        # 读取深圳的订单 CSV 文件
        with open('./data/shenzhen_orders_0324.csv','r',encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                project_id=row['project_id']
                station_id=row['station_id']
                
                if project_id not in project_station_interaction:
                    project_station_interaction[project_id]={}
                    
                if station_id not in project_station_interaction[project_id]:

                    project_loc = self.projects[project_id]
                    station_loc = self.stations[station_id].coord
                    
                    project_station_interaction[project_id][station_id]=self.distance_between_grids(project_loc[0],project_loc[1],station_loc[0],station_loc[1])
                
        self.interaction = project_station_interaction
        
        
    def init_interaction2(self):
        
        project_station_interaction = {}
        
        for project_id, project_loc in self.projects.items():
            project_station_interaction[project_id] = {}
            
            for station_id, station in self.stations.items():
                station_loc = station.coord
                distance = self.distance_between_grids(project_loc[0], project_loc[1], station_loc[0], station_loc[1])
                
                if distance >30:
                    continue
                # 只记录距离小于30km的厂站
                project_station_interaction[project_id][station_id] = distance
                
        self.interaction = project_station_interaction
                
                
                
        
        
        
        
        
        
        
        
        
                    