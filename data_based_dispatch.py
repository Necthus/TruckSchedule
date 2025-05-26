import pandas as pd
from component import *



class RawDataDispatch:
    def __init__(self) -> None:
        # 读取 CSV 文件
        df = pd.read_csv('./data/shenzhen_orders_0324.csv', dtype=str)

        # 提取 order_id 和 station_id，转换为字典
        self.order_station_dict = df.set_index('order_id')['station_id'].to_dict()

    def return_dispatch_list(self,orders:List[Order]):
    
        dispatch_list :List[Dispatch] = []
        
        for order in orders:
            order_id = order.oid
            from_sid = self.order_station_dict[order_id]
            
            for _ in range(order.n_need):
                new_dispatch = Dispatch(order_id,order.pid,from_sid,order.plan_arrive_time-datetime.timedelta(hours=0),from_sid)
                dispatch_list.append(new_dispatch)
            
            
        dispatch_list.sort(key=lambda x: x.dispatch_time)
        return dispatch_list
    
    def external_method_for_random_need(self,current_order:Order):
    
        dispatch_list = []
        
        oid = current_order.oid
        
        from_sid = self.order_station_dict[oid]
        
        truck_need = current_order.n_need
        
        for i in range(truck_need):
            d = Dispatch(oid,current_order.pid,from_sid,current_order.plan_arrive_time,from_sid)
            dispatch_list.append(d)    
        
        return dispatch_list


    

