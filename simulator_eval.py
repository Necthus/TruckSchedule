
import datetime
from utils import *
from component import *


class Environment:
    
    
    def __init__(self,truck_per_station =10,line_size = 1,running_day = 1,start_date = datetime.datetime(2024,5,1)) -> None:
        self.shenzhen_range = [113.770,114.334,22.47,22.803]
        
        self.start_date = start_date
        self.end_date = start_date+datetime.timedelta(days=running_day)
        
        self.current_time = self.start_date
        
        self.load_concrete_time = 10
        self.unload_concrete_time =15
        self.truck_speed = 0.25
        
        self.width, self.height = calculate_range_dimensions(self.shenzhen_range)
        
        self.size_per_grid = 0.5 # km
        self.truck_per_station = truck_per_station
        self.line_size = line_size
        
        self.projects = {}
        self.stations :dict[str,Station]= {}
        self.orders = []
        
        
        self.unscheduled_orders:dict[str,Order] = {}
        self.unfinished_orders:dict[str,Order] = {}
        
        self.finished_orders:dict[str,Order] = {}
        
        self.interaction = {}
        self.init_project()
        self.init_station()
        self.init_orders()
        self.init_interaction()
        
        self.order_iter = OrderIterator(self.orders)
        
        self.truck_list = TruckList()
        
        
        
        self.reward_in_a_step = 0
        self.dispatch_out_in_a_step = 0
        self.total_distance = 0
        
        self.log_file = './running.log'
        
        
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
        
    def count_time_per_fang(self):
    
        total_q = 0
        total_dt = 0
        
        for oid,o in self.finished_orders.items():
            q = o.quantity
            
            dt = round((o.finish_time-o.create_time).total_seconds()/60)
            
            total_q+=q
            total_dt+=dt
            
        return total_dt/total_q    
        
    
    def log(self,msg):
        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write(f'{self.current_time}: {msg}\n')
            file.flush()
        
    def finish_order_once(self,oid):
        
        order = self.unfinished_orders[oid]
        
        order.n_finish+=1
        # 如果已经运输完成了指定次数
        if order.n_finish == order.n_need:
            # 存档
            self.finished_orders[oid]=order
            order.finish_time = self.current_time
            # 从未完成的里面删除
            del self.unfinished_orders[oid]
        
    
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
        
        
    def take_action(self,return_sid,truck:Truck):
            
        truck.ret_sid = return_sid
        truck.left_time = round(self.interaction[truck.to_pid][return_sid]/self.truck_speed)
        
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
        
        
        dispatch_count = s.count_dispatch_given_range(self.current_time,12*60)
        truck_num = s.truck_num
        dist = self.interaction[pid][sid]
        
        return [dispatch_count,truck_num,dist,s.get_next_return_time()]
                        
            
    def schedule_order_no_reward(self,order:Order):
        
        station_list = [[sid,dist] for sid,dist in self.interaction[order.pid].items()]
        
        sid_go_time_dict = {}
        sid_total_time_dict = {}
        can_dispatch_set = set()
          
        for sid,dist in station_list:
            wt = self.stations[sid].product_line.wait_time()
            gt = round(dist/self.truck_speed)
            
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
            
        
    def schedule_order(self,order:Order):
        
        station_list = [[sid,dist] for sid,dist in self.interaction[order.pid].items()]
        
        
        
        sid_go_time_dict = {}
        sid_total_time_dict = {}
        can_dispatch_set = set()
          
        for sid,dist in station_list:
            wt = self.stations[sid].product_line.wait_time()
            gt = round(dist/self.truck_speed)
            
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
        
    def truck_state_change(self,truck:Truck):
        # 1 去厂站装水泥 2 排队等待（不一定）+装水泥（一定时间） 3 去工地卸水泥 + 卸水泥（不需要排队） 4 返程
        
        if truck.state == TruckState.Free:
            print('Error',self.current_time)
            
        elif truck.state == TruckState.Load:
            
            truck.state = TruckState.Transport
            dist = self.interaction[truck.to_pid][truck.from_sid]    
            truck.left_time = round(dist/self.truck_speed)    
            
        elif truck.state == TruckState.Transport:
            truck.state = TruckState.Cast
            truck.left_time = self.unload_concrete_time
            # 到达时将记录走过了多少距离
            self.total_distance+=self.interaction[truck.to_pid][truck.from_sid]
            
            
        elif truck.state == TruckState.Cast:
            truck.state = TruckState.Return
               
            
        elif truck.state == TruckState.Return:
            
            
            truck.state = TruckState.Free
            truck.left_time = 0
            # 回到厂站时将记录走过了多少距离
            self.total_distance+=self.interaction[truck.to_pid][truck.ret_sid]
            
            oid_remove_from_unscheduled = []
            for oid,o in self.unscheduled_orders.items():
                ret = self.schedule_order(o)
                if ret == True:
                    # 从unscheduled里面删去
                    oid_remove_from_unscheduled.append(oid)
                    
            for oid in oid_remove_from_unscheduled:
                del self.unscheduled_orders[oid]
                
            
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
        
    def init_station(self,filename = './shenzhen_stations.csv'):
        
        stations = pd.read_csv(filename,dtype={'id': str})
        
        for index, row in stations.iterrows():
            stationid = row['id']
            x,y = self.lat_lng_to_grid(row['lat'],row['lng'])
            self.stations[stationid]=Station(stationid,self.truck_per_station,self.line_size,(x,y))
        
        
    def init_project(self,filename = './shenzhen_projects.csv'):
        
        projects = pd.read_csv(filename,dtype={'id': str})
        
        for index, row in projects.iterrows():
            id = row['id']
            x,y = self.lat_lng_to_grid(row['lat'],row['lng'])
            self.projects[id]=(x,y)
            
    def init_orders(self,filename = './shenzhen_orders_sorted.csv'):
        fields_to_keep = ['project_id','create_time','deliver_time','order_quantity','ticket_count']
        type_conversions = {'project_id': str,'order_quantity': float,'ticket_count':int}
        df = pd.read_csv(filename, usecols=fields_to_keep, dtype=type_conversions,parse_dates=['create_time','deliver_time'])
        
        # 将 Timestamp 转换为 datetime
        # 将 deliver_time 列转换为 datetime 类型
        df['deliver_time'] = pd.to_datetime(df['deliver_time'])
        df['create_time'] = pd.to_datetime(df['create_time'])
        
        filtered_df = df[(df['create_time'] >= self.start_date) & (df['create_time'] <= self.end_date)]
        
        self.orders = filtered_df.to_dict(orient='records')
        for index, order in enumerate(self.orders, start=1):
            order['id'] = index
        
    def distance_between_grids(self,x1,y1,x2,y2):
        return (abs(x1-x2)+abs(y1-y2))*self.size_per_grid    
    
    
    
    def init_interaction(self):
        import csv
        project_station_interaction = {}
        # 读取深圳的订单 CSV 文件
        with open('shenzhen_orders_sorted.csv','r',encoding='utf-8') as f:
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
                    