import numpy as np
from component import *
from simulator import Environment
from parameter import *
from perception import PerceptionLayer
from toolkit.time import print_with_time
import datetime

from agent.dispatch.base_agent import BaseDispatchAgent
from agent.dispatch.fastest_agent import FastestDispatchAgent
from agent.dispatch.follow_agent import FollowDispatchAgent

from agent.reposition.base_agent import BaseRepositionAgent
from agent.reposition.urgent_agent import UrgentRepositionAgent
from agent.reposition.retrace_agent import RetraceRepositionAgent
from agent.reposition.rl_agent import RLRepositionAgent
from agent.reposition.dispatch_aware_rl_agent import DispatchAwareRLRepositionAgent


def get_all_dates(start_date, end_date):
    """获取起止日期之间的所有日期"""
    dates = []
    current = start_date.date()
    end = end_date.date()
    while current <= end:
        dates.append(current)
        current += datetime.timedelta(days=1)
    return dates


def split_train_test_dates(all_dates, split_ratio):
    """按比例分割训练集和测试集日期"""
    split_idx = int(len(all_dates) * split_ratio)
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]
    return train_dates, test_dates


def create_dispatch_agent():
    """根据参数创建Dispatch Agent"""
    if DISPATCH_METHOD == 'Fastest':
        return FastestDispatchAgent()
    elif DISPATCH_METHOD == 'Follow':
        return FollowDispatchAgent()
    else:
        raise ValueError(f'Unknown DISPATCH_METHOD: {DISPATCH_METHOD}')


def create_reposition_agent():
    """根据参数创建Reposition Agent"""
    if REPOSITION_METHOD == 'Retrace':
        return RetraceRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    elif REPOSITION_METHOD == 'Urgent':
        return UrgentRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    elif REPOSITION_METHOD == 'RL':
        return RLRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    elif REPOSITION_METHOD == 'DispatchAwareRL':
        return DispatchAwareRLRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    else:
        raise ValueError(f'Unknown REPOSITION_METHOD: {REPOSITION_METHOD}')


def run_one_day(env: Environment, chosen_day, dispatch_agent: BaseDispatchAgent,
                reposition_agent: BaseRepositionAgent, perception: PerceptionLayer):
    """模拟运行一天，返回当天指标"""
    env.reset(chosen_day)

    # 感知层每天重置
    perception.reset()

    # RL agent的episode初始化
    if isinstance(reposition_agent, (RLRepositionAgent, DispatchAwareRLRepositionAgent)):
        reposition_agent.before_every_episode()

    while env.truck_iter or env.dispatch_iter or env.order_iter or env.unsolved_dispatch:

        if not env.truck_iter and not env.dispatch_iter and not env.order_iter and env.unsolved_dispatch:
            env.resolve_unsolved_dispatches()
            continue

        event_lt = env.truck_iter.next_event_lt()
        dispatch_lt = env.dispatch_iter.next_dispatch_lt(env.current_time)
        order_lt = env.order_iter.next_order_lt(env.current_time)

        IF_EVENT = 0
        IF_DISPATCH = 1
        IF_ORDER = 2

        lt_list = [event_lt, dispatch_lt, order_lt]
        index = np.argmin(lt_list)
        lt = lt_list[index]

        env.time_pass(lt)

        if index == IF_EVENT:
            current_truck = env.truck_iter.return_next_event()
            assert current_truck.left_time == 0

            env.truck_state_change(current_truck)

            if current_truck.state == TruckState.Return:
                env.finish_order_once(current_truck.oid)
                current_pid = current_truck.to_pid
                available_sids = list(env.interaction[current_pid].keys())

                # 使用RL agent或传统agent进行reposition
                decide_ret_sid = reposition_agent.select_reposition_station(
                    current_pid, current_truck, available_sids, env)

                # 记录reposition到感知层
                repo_record = perception.record_reposition(
                    env.current_time, current_pid, decide_ret_sid, available_sids)

                env.do_return(decide_ret_sid, current_truck)

            if not current_truck.state == TruckState.Free:
                env.truck_iter.insert_in_order(current_truck)
            elif current_truck.state == TruckState.Free:
                del current_truck

        elif index == IF_DISPATCH:
            current_dispatch = env.dispatch_iter.return_next_dispatch()
            serve_oid = current_dispatch.oid

            if serve_oid in env.fail_oid_set:
                continue

            ret = env.execute_single_dispatch(current_dispatch)

            if ret == False:
                env.fail_dispatch += 1
                env.unsolved_dispatch.append(current_dispatch)

        elif index == IF_ORDER:
            current_order = env.order_iter.return_next_order()

            env.unfinished_orders[current_order.oid] = current_order
            env.not_fully_dispatch_orders[current_order.oid] = current_order

            available_sids = list(env.interaction[current_order.pid].keys())
            available_sids = [sid for sid in available_sids if sid in env.open_sid_set]

            for i in range(current_order.n_need):
                # 获取原始意图和实际选择
                intent_sid = dispatch_agent.get_intent_station(current_order, available_sids, env)
                select_sid = dispatch_agent.select_a_station(current_order, available_sids, env)

                # 记录到感知层
                selected_station = env.stations.get(select_sid)
                if selected_station is not None:
                    dispatch_record = perception.record_dispatch(
                        env.current_time, intent_sid, select_sid, selected_station)

                    # 应用感知层奖励（对于DispatchAwareRL agent）
                    if isinstance(reposition_agent, DispatchAwareRLRepositionAgent):
                        reposition_agent.apply_perception_rewards(perception, env)

                new_dispatch = Dispatch(current_order.oid, current_order.pid, select_sid, env.current_time)
                new_dispatch.is_first_dispatch = (i == 0)

                current_dispatch = new_dispatch
                serve_oid = current_dispatch.oid

                if serve_oid in env.fail_oid_set:
                    continue

                ret = env.execute_single_dispatch(current_dispatch)

                if ret == False:
                    env.fail_dispatch += 1
                    env.unsolved_dispatch.append(current_dispatch)

    env.record_site_overtime()

    # RL agent的episode收尾
    if isinstance(reposition_agent, (RLRepositionAgent, DispatchAwareRLRepositionAgent)):
        reposition_agent.after_every_episode(env)


def main():
    env = Environment()
    env.reset_statistic()

    print_with_time("---------------------------实验开始------------------------------")
    print_with_time(args)

    # 创建Agent
    dispatch_agent = create_dispatch_agent()
    reposition_agent = create_reposition_agent()

    # 感知层
    perception = PerceptionLayer()

    print_with_time(f'厂站初始车数量：{STATION_INIT_TRUCK_NUM}，生产线数量：{STATION_PRODUCTION_LINE_SIZE}')
    print_with_time(f"Dispatch method: {DISPATCH_METHOD}, Reposition method: {REPOSITION_METHOD}")

    # 获取所有日期并分割
    all_dates = get_all_dates(EXPERIMENT_START_DATE, EXPERIMENT_END_DATE)
    train_dates, test_dates = split_train_test_dates(all_dates, TRAIN_TEST_SPLIT_RATIO)

    print_with_time(f"总日期数: {len(all_dates)}, 训练日期: {len(train_dates)}, 测试日期: {len(test_dates)}")
    print_with_time(f"训练集日期范围: {train_dates[0]} ~ {train_dates[-1]}" if train_dates else "无训练集")
    print_with_time(f"测试集日期范围: {test_dates[0]} ~ {test_dates[-1]}" if test_dates else "无测试集")

    # 模型初始化
    begin_episode = reposition_agent.model_initialization()
    if begin_episode == 0:
        begin_episode = 0

    # 确定训练/测试模式
    is_train = REPOSITION_TRAIN_MODE or DISPATCH_TRAIN_MODE

    if is_train:
        print_with_time('---------------------------训练模式------------------------------')
        episode_num = REPOSITION_EPISODE_NUM
        use_dates = train_dates

        for i_e in range(begin_episode, episode_num):
            env.i_episode = i_e
            chosen_day = use_dates[i_e % len(use_dates)]
            print_with_time(f'\n[训练 Episode {i_e}] 日期: {chosen_day}')
            run_one_day(env, chosen_day, dispatch_agent, reposition_agent, perception)

        print_with_time(f'\n---------------------------训练完成------------------------------')

    else:
        print_with_time('---------------------------测试模式------------------------------')
        use_dates = all_dates
        for test_idx in range(TEST_REPEAT_NUM):
            for day_idx, chosen_day in enumerate(use_dates):
                print_with_time(f'\n[测试 第{test_idx + 1}轮 第{day_idx + 1}天] 日期: {chosen_day}')
                run_one_day(env, chosen_day, dispatch_agent, reposition_agent, perception)

    env.print_result()


if __name__ == '__main__':
    main()
