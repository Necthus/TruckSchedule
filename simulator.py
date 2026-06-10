
import datetime
import bisect
from toolkit.geocal import *
from component import *
import numpy as np
import csv

from parameter import *

class Environment:


    def __init__(self) -> None:

        self.i_episode = 0  # 当前episode编号

        self.projects :dict[str,Project]= {}
        self.stations :dict[str,Station]= {}
        self.orders:List[Order]= []


        '''两个字典，以日期为键存储着每天的即时订单和计划订单'''
        self.plan_orders_dict_by_date :dict[datetime.date,List[Order]]= {}
        self.instant_orders_dict_by_date :dict[datetime.date,List[Order]]= {}


        self.interaction = {}

        # key是一个元组，代表(s1,s2)
        self.station_dist_matrix = {}

        self.origin_cooperation:dict[str,set] = {}
        self.origin_dispatch:dict = {}

        self.init_station()

        self.init_project()
        self.init_orders()
        self.divide_orders()

        self.init_interaction()

        self.fail_order = 0

        self.must_rep = 0

        self.fail_oid_set = set()




    def derive_plan_orders_dispatches(self,plan_orders:List[Order]):

        dispatches = []



        for order in plan_orders:
            # 订单的工地
            oid = order.oid

            pid = order.pid
            # 订单需要的车数
            truck_need = order.n_need

            chosen_station = self.origin_dispatch[oid]

            order_time = order.plan_arrive_time

            estimated_travel_time = round(self.interaction[pid][chosen_station]/TRUCK_SPEED)

            # manual_advance_time = datetime.timedelta(minutes=30)
            manual_advance_time = datetime.timedelta(minutes=(estimated_travel_time+LOAD_CONCRETE_TIME))

            for i in range(truck_need):
                dispatch_time = order_time - manual_advance_time
                new_dispatch = Dispatch(order.oid, order.pid, chosen_station, dispatch_time)
                new_dispatch.is_first_dispatch = (i == 0)
                dispatches.append(new_dispatch)

        return dispatches



    def reset(self, day:datetime.date):

        self.current_time = datetime.datetime.combine(day, datetime.time(0, 0))

        self.unsolved_dispatch = []
        self.project_lastest_connect_station:dict[str,str] = {}

        for sid,s in self.stations.items():

            s.reset()

        for pid,p in self.projects.items():
            p.last_working_time = None

        # 这个指的是未完全调度的订单，即还需要派出车辆
        self.not_fully_dispatch_orders:dict[str,Order] = {} # 是下面dict的子集
        # 这个指的是未完成的订单，还没有完全浇筑完，可能车辆已经全部派出，也可能未完全派出。
        self.unfinished_orders:dict[str,Order] = {}
        # 完成的订单进行存档
        self.finished_orders:dict[str,Order] = {}

        if IGNORE_INSTANT_ORDERS:
            instant_orders = []
        else:
            instant_orders = copy.deepcopy(self.instant_orders_dict_by_date.get(day, []))
        plan_orders = copy.deepcopy(self.plan_orders_dict_by_date.get(day, []))

        for order in plan_orders:
            self.unfinished_orders[order.oid]=order
            self.not_fully_dispatch_orders[order.oid]=order


        # 先处理计划订单，使用preplan agent
        # -----------------------------------------------------------

        # 获取对计划的dispatch结果
        plan_orders_dispatches :List[Dispatch] = self.derive_plan_orders_dispatches(plan_orders)

        # 将每个dispatch的预定时间写入对应厂站的arrange_dispatch
        for dispatch in plan_orders_dispatches:
            sid = dispatch.from_sid
            bisect.insort(self.stations[sid].arrange_dispatch, dispatch.dispatch_time)

        self.open_sid_set = set()


        for dispatch in plan_orders_dispatches:
            self.open_sid_set.add(dispatch.from_sid)

        # 剩下的 instant_orders，留着后面遍历处理
        self.truck_iter = TruckIterator()
        self.dispatch_iter = DispatchIterator(plan_orders_dispatches)
        self.order_iter = OrderIterator(instant_orders)

        # 以下都是指标
        # self.reward_in_a_step = 0
        # self.dispatch_out_in_a_step = 0


        self.plan_real_arrive_time_compare = []
        self.cast_interval_list = []



        self.workover_reposition_record = []



    def reset_statistic(self):

        self.fail_dispatch = 0

        self.total_delivered_quantity = 0
        self.revenue = 0

        self.go_distance = 0
        self.return_distance = 0
        self.fuel_consumption = 0

        self.discontinuity_penalty = 0
        self.overtime_penalty = 0

        self.station_overtime_cost = 0
        self.project_overtime_cost = 0


    def time_pass(self, dt_in_minute):
        self.truck_iter.pass_time(dt_in_minute)
        for id,station in self.stations.items():
            station.step_time(dt_in_minute)
        self.current_time+=datetime.timedelta(minutes=dt_in_minute)


    def reset_stations(self):
        for sid,s in self.stations.items():
            s.reset()

    def change_truck_num(self,truck_num):
        for sid,s in self.stations.items():
            s.truck_num = truck_num


    def finish_order_once(self,oid):

        # 如果是已经失败的订单，就别管了
        if oid in self.fail_oid_set:
            return

        order = self.unfinished_orders[oid]

        order.n_already_finished+=1

        order.last_cast_time = self.current_time


        # 如果已经运输完成了指定次数
        if order.n_already_finished == order.n_need:
            # 存档
            self.finished_orders[oid]=order
            order.finish_time = self.current_time
            # 从未完成的里面删除
            del self.unfinished_orders[oid]

            # 利润获取
            self.revenue+=order.revenue
            self.total_delivered_quantity+=order.quantity

    # def make_order_fail(self,oid):

    #     del self.unfinished_orders[oid]
    #     if oid in self.not_fully_dispatch_orders:
    #         del self.not_fully_dispatch_orders[oid]

    #     self.fail_oid_set.add(oid)



    def verify_dispatchs(self,dispatchs:List[Dispatch]):

        true_need_dict = {}
        dispatch_dict = {}


        for k,v in self.not_fully_dispatch_orders.items():

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


    def execute_single_dispatch(self, dispatch: Dispatch):

        oid = dispatch.oid

        new_truck = self.stations[dispatch.from_sid].dispatch_truck(oid, dispatch.pid, self.current_time, dispatch.is_first_dispatch)

        if new_truck is None:
            return False
        else:
            new_truck.ret_sid = dispatch.from_sid
            self.truck_iter.insert_in_order(new_truck)
            dispatch.real_dispatch_time = self.current_time
            self.not_fully_dispatch_orders[oid].n_already_dispatched += 1
            assert self.not_fully_dispatch_orders[oid].n_already_dispatched <= self.not_fully_dispatch_orders[oid].n_need
            self.project_lastest_connect_station[dispatch.pid] = dispatch.from_sid

            if self.not_fully_dispatch_orders[oid].n_already_dispatched == self.not_fully_dispatch_orders[oid].n_need:
                del self.not_fully_dispatch_orders[oid]

            return True

    def execute_unsolved_dispatch(self):

        still_unfinished_dispatch = []
        # 遍历未完成的dispatch
        for d in self.unsolved_dispatch:
            # 执行一个dispatch
            ret = self.execute_single_dispatch(d)
            # 仍然未完成
            if ret == False:
                still_unfinished_dispatch.append(d)

        # 重新赋值未完成的dispatch
        self.unsolved_dispatch = still_unfinished_dispatch

    def resolve_unsolved_dispatches(self):
        """当三个事件迭代器都为空时，尝试从附近厂站调车来解决未完成的dispatch"""

        if not self.unsolved_dispatch:
            return

        needed_trucks = {}
        for d in self.unsolved_dispatch:
            sid = d.from_sid
            needed_trucks[sid] = needed_trucks.get(sid, 0) + 1

        max_transfer_time = 0

        for sid, need_count in needed_trucks.items():
            s = self.stations[sid]
            shortage = need_count - s.truck_num
            if shortage <= 0:
                continue

            candidates = []
            for other_sid, other_s in self.stations.items():
                if other_sid == sid:
                    continue
                other_need = needed_trucks.get(other_sid, 0)
                available = other_s.truck_num - other_need
                if available > 0:
                    dist = self.station_dist_matrix[(sid, other_sid)]
                    travel_time = round(dist / TRUCK_SPEED)
                    candidates.append((travel_time, dist, other_sid, other_s, available))

            candidates.sort(key=lambda x: x[0])

            still_need = shortage
            for travel_time, dist, other_sid, other_s, available in candidates:
                if still_need <= 0:
                    break
                transfer_count = min(still_need, available)
                still_need -= transfer_count

                other_s.truck_num -= transfer_count
                s.truck_num += transfer_count

                max_transfer_time = max(max_transfer_time, travel_time)

        if max_transfer_time > 0:
            self.time_pass(max_transfer_time)

        self.execute_unsolved_dispatch()

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



    def do_return(self,return_sid,truck:Truck):

        # 设定搅拌车的返回sid
        truck.ret_sid = return_sid
        # 设定返回时间
        truck.left_time = round(self.interaction[truck.to_pid][return_sid]/TRUCK_SPEED)
        # 通知厂站有一个车要回来了
        self.stations[return_sid].add_return_truck(truck.left_time)


    def get_action_feature(self,sid,pid):

        s = self.stations[sid]


        dispatch_count = s.count_dispatch_given_range(self.current_time,60)
        truck_num = s.truck_num
        dist = self.interaction[pid][sid]
        # 特征：过去1小时发车数量，当前车数量，距离工地距离，
        future_dispatch_feature = []

        for trange in [30,60,120]:
            future_dispatch_feature.append(s.count_future_dispatch_given_range(self.current_time,trange))


        returning_count = len(s.return_time_list)

        return [dispatch_count,truck_num,dist,s.get_next_return_time(),returning_count]+future_dispatch_feature

    def truck_state_change(self,truck:Truck):
        # 1 去厂站装水泥 2 排队等待（不一定）+装水泥（一定时间） 3 去工地卸水泥 + 卸水泥（不需要排队） 4 返程
        # 车辆从闲置状态开始/不可能

        assert truck.left_time == 0

        if truck.state == TruckState.Free:

            print('Error',self.current_time)
            raise ValueError('Truck is free, but not in the station.')

        # 车辆结束装载状态
        elif truck.state == TruckState.Load:

            truck.state = TruckState.Transport
            # 从厂站的loading队列中移除
            loading_station = self.stations[truck.from_sid]
            if truck in loading_station.loading_trucks:
                loading_station.loading_trucks.remove(truck)
            dist = self.interaction[truck.to_pid][truck.from_sid]
            truck.left_time = round(dist/TRUCK_SPEED)


        # 车辆结束运输阶段，已经到达了工地
        elif truck.state == TruckState.Transport:
            # 车辆服务的订单
            oid = truck.oid
            # 看看距离
            distance = self.interaction[truck.to_pid][truck.from_sid]

            self.go_distance+=distance


            # 变为浇筑状态
            truck.state = TruckState.Cast


            current_order = self.unfinished_orders[oid]


            # 如果还一次都没有浇筑，说明是第一个到达的车
            if current_order.whether_arrive==False:

                current_order.whether_arrive = True

                plan_time = current_order.plan_arrive_time
                current_time = self.current_time
                # 如果提早到了，计算等待的惩罚油钱

                dt = current_time-plan_time
                dt_in_minute = dt.total_seconds()/60


                if dt_in_minute<0:
                    truck.left_time += -dt_in_minute

                    self.fuel_consumption+=  (-dt_in_minute)*LOAD_WAITING_FUEL_CONSUMPTION_PER_MINUTE


                else:
                    if dt_in_minute>6*60:
                        self.overtime_penalty+=20000
                    elif dt_in_minute>2*60:
                        self.overtime_penalty+=0.01*current_order.revenue

                self.plan_real_arrive_time_compare.append((current_order.oid,plan_time,current_time,dt_in_minute))



            # 如果已经浇筑过了，说明不是第一个到达的
            else:
                # 上一辆车还没浇筑完，不存在连续浇筑惩罚
                # 现在应该不存在这个情况
                if current_order.last_arrive_time == None:
                    dt_in_minute = 0
                else:
                    dt = self.current_time-current_order.last_arrive_time
                    dt_in_minute = dt.total_seconds()/60

                if dt_in_minute>UNLOAD_CONCRETE_TIME+60:
                    self.discontinuity_penalty+=5000

                self.cast_interval_list.append((current_order.oid,self.current_time,dt_in_minute))
            current_order.last_arrive_time = self.current_time

            truck.left_time += UNLOAD_CONCRETE_TIME


        # 车辆浇筑完了，要返回了
        elif truck.state == TruckState.Cast:

            truck.state = TruckState.Return
            # 车辆离开工地，记录工地的最后工作时间
            self.projects[truck.to_pid].last_working_time = self.current_time

            # 将在另外的地方写入返回所需要的·时间

        # 返回结束了
        elif truck.state == TruckState.Return:

            # 车辆回到厂站，变为可用状态
            truck.state = TruckState.Free
            truck.left_time = 0

            sid = truck.ret_sid
            s = self.stations[sid]
            s.receive_truck(truck)
            s.last_working_time = self.current_time


            # 回到厂站时将记录走过了多少距离
            distance = self.interaction[truck.to_pid][truck.ret_sid]

            self.return_distance+=distance

            # 回到厂站后，由于厂站有车，所以执行剩余的dispatch
            self.execute_unsolved_dispatch()



    def init_station(self,filename = './data/shenzhen_stations.csv'):

        stations_coord_dict :dict[str,tuple]= {}

        with open(filename,'r',encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stationid = row['id']
                lon,lat = float(row['lng']),float(row['lat'])

                stations_coord_dict[stationid]=(lon,lat)

        self.stations=delete_duplicate_stations(stations_coord_dict)

        for sid1,s1 in self.stations.items():
            for sid2,s2 in self.stations.items():
                key1 = (sid1,sid2)
                key2 = (sid2,sid1)
                if key1 in self.station_dist_matrix:
                    continue

                if sid1==sid2:
                    dist =0
                else:
                    dist = self.distance_between_grids(s1.x,s1.y,s2.x,s2.y)

                self.station_dist_matrix[key1] = dist
                self.station_dist_matrix[key2] = dist


    def init_project(self,filename = './data/shenzhen_projects.csv'):


        with open(filename,'r',encoding='utf-8') as f:

            reader = csv.DictReader(f)

            new_id = 0


            for row in reader:
                id = row['id']
                name = row['name']
                lon,lat = float(row['lng']),float(row['lat'])

                new_id_str = f'P{new_id}'
                new_id+=1
                self.projects[new_id_str]=Project(new_id_str,name,(lon,lat))
                Project.id_map[id]=new_id_str




    def init_orders(self,filename = './data/shenzhen_orders_0324.csv'):

        new_id = 0


        with open(filename,'r',encoding='utf-8') as f:

            reader = csv.DictReader(f)

            for row in reader:
                project_id = row['project_id']

                project_id = Project.id_map[project_id]

                deliver_time = datetime.datetime.strptime(row['deliver_time'],'%Y-%m-%d %H:%M:%S')
                order_quantity = float(row['order_quantity'])
                ticket_count = int(row['ticket_count'])
                pouring_type = row['pouring_type']


                if self.check_valid_order_time(deliver_time):

                    new_id_str = f'O{new_id}'
                    new_id += 1

                    self.orders.append(Order(new_id_str,project_id,order_quantity,ticket_count,pouring_type,deliver_time))

                    original_sid = row['station_id']
                    new_sid = Station.id_map[original_sid]
                    self.origin_dispatch[new_id_str]=new_sid

                    if project_id not in self.origin_cooperation:
                        self.origin_cooperation[project_id]=set()

                    self.origin_cooperation[project_id].add(new_sid)

    def check_valid_order_time(self,time:datetime.datetime):

        if time.hour>=ORDER_START_HOUR and time.hour<ORDER_END_HOUR and \
            time.date()>=EXPERIMENT_START_DATE.date() and time.date()<=EXPERIMENT_END_DATE.date():
            return True
        else:
            return False

    def divide_orders(self):

        for order in self.orders:

            pouring_type = order.pouring_type

            if pouring_type == '自卸':
                this_dict = self.instant_orders_dict_by_date
            else:

                this_dict = self.plan_orders_dict_by_date

            order_date = order.plan_arrive_time.date()

            if order_date not in this_dict:
                this_dict[order_date]=[]

            this_dict[order_date].append(order)


    # 有个run_given_day，放到backup中了

    def record_site_overtime(self):
        """每天结束后，计算各厂站和各工地超过ORDER_END_HOUR工作的加班补偿"""

        # 厂站加班
        for sid, station in self.stations.items():
            if station.last_working_time is not None:
                end_of_day = datetime.datetime.combine(
                    station.last_working_time.date(),
                    datetime.time(ORDER_END_HOUR, 0)
                )
                if station.last_working_time > end_of_day:
                    overtime_minutes = (station.last_working_time - end_of_day).total_seconds() / 60
                    self.station_overtime_cost += overtime_minutes * OVERTIME_PAY_FOR_STATION_PER_MINUTE

        # 工地加班
        for pid, project in self.projects.items():
            if project.last_working_time is not None:
                end_of_day = datetime.datetime.combine(
                    project.last_working_time.date(),
                    datetime.time(ORDER_END_HOUR, 0)
                )
                if project.last_working_time > end_of_day:
                    overtime_minutes = (project.last_working_time - end_of_day).total_seconds() / 60
                    self.project_overtime_cost += overtime_minutes * OVERTIME_PAY_FOR_PROJECT_PER_MINUTE


    def distance_between_grids(self,x1,y1,x2,y2):
        return (abs(x1-x2)+abs(y1-y2))*GRID_SIZE_KM


    def print_with_system_time(self, message):
        current_system_time = self.current_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{current_system_time}] {message}')

    def init_interaction(self):

        project_station_interaction = {}

        for project_id, project in self.projects.items():

            project_loc = (project.x,project.y)
            project_station_interaction[project_id] = {}

            if project_id in self.origin_cooperation:
                origin_coop = self.origin_cooperation[project_id]
            else:
                origin_coop = set()


            for station_id, station in self.stations.items():
                station_loc = (station.x,station.y)
                distance = self.distance_between_grids(project_loc[0], project_loc[1], station_loc[0], station_loc[1])

                if distance >30 and station_id not in origin_coop:
                    continue
                # 只记录距离小于30km的厂站，或者初始合作的厂站
                project_station_interaction[project_id][station_id] = distance


        self.interaction = project_station_interaction


    def calculate_metrics(self):



        print('以下是路程指标：')
        print(f'总的过去距离：{self.go_distance} km')
        print(f'总的返回距离：{self.return_distance} km')

        self.fuel_consumption += self.go_distance * LOAD_GO_FUEL_CONSUMPTION_PER_KM + self.return_distance * EMPTY_RETURN_FUEL_CONSUMPTION_PER_KM
        print(f'总的燃油消耗：{self.fuel_consumption} L')

        cost = self.fuel_consumption*OIL_PRICE

        print(f'总的燃油成本：{cost} RMB')


        print(f'总的交付方量：{self.total_delivered_quantity} m3')
        print(f'总的交付收入：{self.revenue} RMB')

        print(f'总的超时罚款：{self.overtime_penalty} RMB')
        print(f'总的连续浇筑罚款：{self.discontinuity_penalty} RMB')
        print(f'总的厂站加班补偿支出：{self.station_overtime_cost:.2f} RMB')
        print(f'总的工地加班补偿支出：{self.project_overtime_cost:.2f} RMB')


        total_cost = cost + self.overtime_penalty + self.discontinuity_penalty + self.station_overtime_cost + self.project_overtime_cost
        if self.revenue > 0:
            percent = total_cost/self.revenue
            print(f'总的成本：{total_cost} RMB，占收入的{percent*100:.2f}%')
        else:
            print(f'总的成本：{total_cost} RMB')

    def print_result(self):

        self.calculate_metrics()

        # # 以下都是指标
        # self.print_with_system_time(f'未完成订单数量：{len(self.unfinished_orders)}')
        # self.print_with_system_time('以下是计划订单的计划到达时间和实际到达时间对比（单位分钟）：')
        # for i in self.plan_real_arrive_time_compare:
        #     self.print_with_system_time(i)

        # self.print_with_system_time('以下是实时订单的连续浇筑时间间隔（单位分钟）：')
        # for i in self.cast_interval_list:
        #     if i[2]>30:

        #         self.print_with_system_time(i)


        # self.print_with_system_time(f'完成的订单：{len(self.finished_orders)}')

        # for oid,order in self.finished_orders.items():
        #     self.print_with_system_time(f'订单{oid}，方量{order.quantity}，计划到达时间{order.plan_arrive_time}，实际完成时间{order.finish_time}')

        # self.print_with_system_time(f'以下是工作结束后所有车回到原厂站的记录：')
        # self.print_with_system_time(f'总共调度回原厂站的车的数量：{len(self.workover_reposition_record)}')

        # for record in self.workover_reposition_record:
        #     self.print_with_system_time(record)






