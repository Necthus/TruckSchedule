import numpy as np
from component import *
from simulator import Environment
from parameter import *



from agent.dispatch.base_agent import BaseDispatchAgent
from agent.dispatch.fastest_agent import FastestDispatchAgent
from agent.dispatch.follow_agent import FollowDispatchAgent

from agent.reposition.base_agent import BaseRepositionAgent
from agent.reposition.urgent_agent import UrgentRepositionAgent
from agent.reposition.retrace_agent import RetraceRepositionAgent



env = Environment()
env.reset_statistic()

print("---------------------------实验开始------------------------------")
print(args)


if DISPATCH_METHOD == 'Fastest':
    dispatch_agent = FastestDispatchAgent()
elif DISPATCH_METHOD == 'Follow':
    dispatch_agent = FollowDispatchAgent()

if REPOSITION_METHOD == 'Retrace':
    reposition_agent = RetraceRepositionAgent()
elif REPOSITION_METHOD == 'Urgent':
    reposition_agent = UrgentRepositionAgent()

print(f'厂站初始车数量：{STATION_INIT_TRUCK_NUM}，生产线数量：{STATION_PRODUCTION_LINE_SIZE}')
print(f"Dispatch method: {DISPATCH_METHOD}, Reposition method: {REPOSITION_METHOD}")


chosen_day = EXPERIMENT_START_DATE.date()

while chosen_day <= EXPERIMENT_END_DATE.date():


    assert chosen_day >= EXPERIMENT_START_DATE.date() and chosen_day <= EXPERIMENT_END_DATE.date()

    env.reset(chosen_day)

    chosen_day += datetime.timedelta(days=1)



    while env.truck_iter or env.dispatch_iter or env.order_iter or env.unsolved_dispatch:

        # 当三个事件迭代器都为空，但仍有未解决的dispatch时，从附近厂站调车
        if not env.truck_iter and not env.dispatch_iter and not env.order_iter and env.unsolved_dispatch:
            env.resolve_unsolved_dispatches()
            continue

        event_lt = env.truck_iter.next_event_lt()
        dispatch_lt = env.dispatch_iter.next_dispatch_lt(env.current_time)
        order_lt = env.order_iter.next_order_lt(env.current_time)

        # 三种事件中最小的时间
        # 0. 车辆事件(状态转换) 1. 发车事件 2. 实时订单事件

        IF_EVENT = 0
        IF_DISPATCH = 1
        IF_ORDER = 2

        lt_list = [event_lt,dispatch_lt,order_lt]
        index = np.argmin(lt_list)
        lt = lt_list[index]

        # 模拟器时间推进
        env.time_pass(lt)

        # 碰到车辆事件了
        if index == IF_EVENT:
            # 车辆状态转换

            current_truck = env.truck_iter.return_next_event()

            assert current_truck.left_time == 0

            env.truck_state_change(current_truck)

            # 需要进行返程调度
            if current_truck.state == TruckState.Return:
                # 订单完成一次
                env.finish_order_once(current_truck.oid)
                # 当前车辆在的工地
                current_pid = current_truck.to_pid
                # 可用返回的厂站
                available_sids = list(env.interaction[current_pid].keys())

                decide_ret_sid = reposition_agent.select_reposition_station(current_pid,current_truck,available_sids,env)

                env.do_return(decide_ret_sid,current_truck)


            # 如果车辆不是变成了闲置状态（已经返回到了厂站），就放回到事件队列里面
            if not current_truck.state == TruckState.Free:
                env.truck_iter.insert_in_order(current_truck)

            elif current_truck.state == TruckState.Free:
                del current_truck

        # 碰到发车dispatch了
        elif index == IF_DISPATCH:
            # 从迭代器里面取出一个dispatch
            current_dispatch = env.dispatch_iter.return_next_dispatch()
            # 首先判断这个订单是不是已经废了，因为超时和浇筑不连续
            serve_oid = current_dispatch.oid

            if serve_oid in env.fail_oid_set:
                # 直接就跳过了
                continue

            # 没问题才继续执行
            # 获取发车的结果
            ret = env.execute_single_dispatch(current_dispatch)
            # 如果发车不成功

            # dispatch 保留only当不成功时，否则直接就内存回收了

            if ret == False:
                # 记为未完成的dispatch
                env.fail_dispatch+=1

                sid = current_dispatch.from_sid
                env.unsolved_dispatch.append(current_dispatch)

        # 碰到实时订单了，需要调度
        elif index == IF_ORDER:
            current_order = env.order_iter.return_next_order()

            env.unfinished_orders[current_order.oid] = current_order
            env.not_fully_dispatch_orders[current_order.oid] = current_order

            available_sids = list(env.interaction[current_order.pid].keys())

            available_sids = [sid for sid in available_sids if sid in env.open_sid_set]


            for i in range(current_order.n_need):

                select_sid = dispatch_agent.select_a_station(current_order, available_sids, env)
                new_dispatch = Dispatch(current_order.oid, current_order.pid, select_sid, env.current_time)
                new_dispatch.is_first_dispatch = (i == 0)

                # 不插入iter了 直接派送
                # env.dispatch_iter.insert(new_dispatch)

                current_dispatch = new_dispatch


                serve_oid = current_dispatch.oid

                if serve_oid in env.fail_oid_set:
                    # 直接就跳过了
                    continue

                # 没问题才继续执行
                # 获取发车的结果
                ret = env.execute_single_dispatch(current_dispatch)
                # 如果发车不成功

                # dispatch 保留only当不成功时，否则直接就内存回收了

                if ret == False:
                    # 记为未完成的dispatch
                    env.fail_dispatch+=1

                    sid = current_dispatch.from_sid
                    env.unsolved_dispatch.append(current_dispatch)



    env.record_site_overtime()

env.print_result()
