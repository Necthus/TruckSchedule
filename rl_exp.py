from reinforce import *
from simulator import *

agent = REINFORCE(1,10,params_path='./result/model_params_0415.pth')
env = Environment(55,4,running_day=1,start_date=datetime.datetime(2024,5,1))


raw_data_dispatch = RawDataDispatch()

dispatch_list_for_fixed_need = raw_data_dispatch.return_dispatch_list(env.unscheduled_orders.values())

env.continuous_running(dispatchs=dispatch_list_for_fixed_need,external_method=raw_data_dispatch.external_method_for_random_need,reposition_method=agent.take_action_max_prob)