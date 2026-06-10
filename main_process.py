import json
import os
import random

import numpy as np
import torch
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
from agent.reposition.scratch_rl_agent import (
    ScratchCombinedRLRepositionAgent,
    ScratchCostOnlyRLRepositionAgent,
    ScratchDispatchAwareNoCostRLRepositionAgent,
    ScratchDispatchAwareNoShapingRLRepositionAgent,
    ScratchDispatchAwareRLRepositionAgent,
    ScratchPerceptionOnlyRLRepositionAgent,
    ScratchRLRepositionAgent,
)


def get_all_dates(start_date, end_date):
    """获取起止日期之间的所有日期"""
    dates = []
    current = start_date.date()
    end = end_date.date()
    while current <= end:
        dates.append(current)
        current += datetime.timedelta(days=1)
    return dates


def split_train_valid_test_dates(all_dates, train_ratio, validation_ratio):
    """按时间顺序分割训练集、验证集和测试集日期。"""
    if not all_dates:
        return [], [], []

    if train_ratio <= 0:
        raise ValueError('TRAIN_TEST_SPLIT_RATIO must be positive')
    if validation_ratio < 0:
        raise ValueError('VALIDATION_SPLIT_RATIO must be non-negative')
    if train_ratio + validation_ratio >= 1:
        raise ValueError('TRAIN_TEST_SPLIT_RATIO + VALIDATION_SPLIT_RATIO must be less than 1')

    train_end = max(1, int(len(all_dates) * train_ratio))
    valid_end = int(len(all_dates) * (train_ratio + validation_ratio))

    if validation_ratio > 0:
        valid_end = max(train_end + 1, valid_end)
    valid_end = min(valid_end, len(all_dates) - 1)

    train_dates = all_dates[:train_end]
    validation_dates = all_dates[train_end:valid_end]
    test_dates = all_dates[valid_end:]
    return train_dates, validation_dates, test_dates


def total_cost(env: Environment):
    fuel_liters = (
        env.fuel_consumption
        + env.go_distance * LOAD_GO_FUEL_CONSUMPTION_PER_KM
        + env.return_distance * EMPTY_RETURN_FUEL_CONSUMPTION_PER_KM
    )
    return (
        fuel_liters * OIL_PRICE
        + env.overtime_penalty
        + env.discontinuity_penalty
        + env.station_overtime_cost
        + env.project_overtime_cost
    )


def collect_metrics(env: Environment):
    fuel_liters = (
        env.fuel_consumption
        + env.go_distance * LOAD_GO_FUEL_CONSUMPTION_PER_KM
        + env.return_distance * EMPTY_RETURN_FUEL_CONSUMPTION_PER_KM
    )
    return {
        'cost': total_cost(env),
        'fuel_cost': fuel_liters * OIL_PRICE,
        'overtime': env.overtime_penalty,
        'discontinuity': env.discontinuity_penalty,
        'station_ot': env.station_overtime_cost,
        'project_ot': env.project_overtime_cost,
        'go_km': env.go_distance,
        'return_km': env.return_distance,
    }


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_training_date(train_dates, episode_num):
    if not SHUFFLE_TRAIN_DATES:
        return train_dates[episode_num % len(train_dates)]

    cycle = episode_num // len(train_dates)
    offset = episode_num % len(train_dates)
    shuffled_dates = train_dates.copy()
    random.Random(SEED + cycle).shuffle(shuffled_dates)
    return shuffled_dates[offset]


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
    elif REPOSITION_METHOD == 'ScratchRL':
        return ScratchRLRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    elif REPOSITION_METHOD == 'ScratchDispatchAwareRL':
        return ScratchDispatchAwareRLRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    elif REPOSITION_METHOD == 'ScratchDispatchAwareNoCostRL':
        return ScratchDispatchAwareNoCostRLRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    elif REPOSITION_METHOD == 'ScratchDispatchAwareNoShapingRL':
        return ScratchDispatchAwareNoShapingRLRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    elif REPOSITION_METHOD == 'ScratchCombinedRL':
        return ScratchCombinedRLRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    elif REPOSITION_METHOD == 'ScratchCostOnlyRL':
        return ScratchCostOnlyRLRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    elif REPOSITION_METHOD == 'ScratchPerceptionOnlyRL':
        return ScratchPerceptionOnlyRLRepositionAgent(train_mode=REPOSITION_TRAIN_MODE)
    else:
        raise ValueError(f'Unknown REPOSITION_METHOD: {REPOSITION_METHOD}')


def evaluate_reposition_agent_on_dates(reposition_agent: BaseRepositionAgent, dates):
    """用当前策略在给定日期上做贪心验证，不更新网络。"""
    eval_env = Environment()
    eval_env.reset_statistic()
    eval_dispatch_agent = create_dispatch_agent()
    eval_perception = PerceptionLayer()
    previous_train_mode = reposition_agent.train_mode

    try:
        reposition_agent.train_mode = False
        for day in dates:
            run_one_day(eval_env, day, eval_dispatch_agent, reposition_agent, eval_perception)
        return collect_metrics(eval_env)
    finally:
        reposition_agent.train_mode = previous_train_mode
        reposition_agent.before_every_episode()


def save_validation_best(reposition_agent: BaseRepositionAgent, episode_num, metrics, validation_dates):
    if not hasattr(reposition_agent, 'algo') or not hasattr(reposition_agent, 'model_save_dir'):
        return

    save_dir = reposition_agent.model_save_dir
    os.makedirs(save_dir, exist_ok=True)
    reposition_agent.algo.save(save_dir, episode_num)

    metadata = {
        'best_epoch': episode_num,
        'validation_cost': metrics['cost'],
        'validation_dates': [str(day) for day in validation_dates],
        'reposition_method': REPOSITION_METHOD,
        'dispatch_method': DISPATCH_METHOD,
    }
    metadata_path = os.path.join(save_dir, 'best_checkpoint.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def run_one_day(env: Environment, chosen_day, dispatch_agent: BaseDispatchAgent,
                reposition_agent: BaseRepositionAgent, perception: PerceptionLayer):
    """模拟运行一天，返回当天指标"""
    env.reset(chosen_day)

    # 感知层每天重置
    perception.reset()

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

            all_available_sids = list(env.interaction[current_order.pid].keys())
            available_sids = [sid for sid in all_available_sids if sid in env.open_sid_set]
            if not available_sids:
                available_sids = all_available_sids

            for i in range(current_order.n_need):
                # 获取原始意图和实际选择
                intent_sid = dispatch_agent.get_intent_station(current_order, available_sids, env)
                select_sid = dispatch_agent.select_a_station(current_order, available_sids, env)

                # 记录到感知层
                selected_station = env.stations.get(select_sid)
                if selected_station is not None:
                    dispatch_record = perception.record_dispatch(
                        env.current_time, intent_sid, select_sid, selected_station)

                    # 应用感知层奖励（对于支持感知奖励的Reposition agent）
                    if hasattr(reposition_agent, 'apply_perception_rewards'):
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

    reposition_agent.after_every_episode(env)


def main():
    set_random_seed(SEED)
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
    train_dates, validation_dates, test_dates = split_train_valid_test_dates(
        all_dates,
        TRAIN_TEST_SPLIT_RATIO,
        VALIDATION_SPLIT_RATIO,
    )

    print_with_time(
        f"总日期数: {len(all_dates)}, 训练日期: {len(train_dates)}, "
        f"验证日期: {len(validation_dates)}, 测试日期: {len(test_dates)}"
    )
    print_with_time(f"训练集日期范围: {train_dates[0]} ~ {train_dates[-1]}" if train_dates else "无训练集")
    print_with_time(
        f"验证集日期范围: {validation_dates[0]} ~ {validation_dates[-1]}"
        if validation_dates else "无验证集"
    )
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
        validation_enabled = bool(validation_dates) and VALIDATION_FREQUENCY > 0
        best_validation_cost = float('inf')
        best_validation_episode = -1
        no_improve_count = 0

        if validation_enabled:
            print_with_time(
                f'训练中验证: 每 {VALIDATION_FREQUENCY} episode 验证一次，'
                f'patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}'
            )
        else:
            print_with_time('训练中验证关闭：无验证集或 VALIDATION_FREQUENCY <= 0')

        print_with_time(f'训练日期打乱: {SHUFFLE_TRAIN_DATES}')

        for i_e in range(begin_episode, episode_num):
            env.i_episode = i_e
            chosen_day = select_training_date(use_dates, i_e)
            print_with_time(f'\n[训练 Episode {i_e}] 日期: {chosen_day}')
            run_one_day(env, chosen_day, dispatch_agent, reposition_agent, perception)

            should_validate = (
                validation_enabled
                and ((i_e + 1) % VALIDATION_FREQUENCY == 0 or i_e + 1 == episode_num)
            )
            if should_validate:
                metrics = evaluate_reposition_agent_on_dates(reposition_agent, validation_dates)
                validation_cost = metrics['cost']
                improved = validation_cost < best_validation_cost - EARLY_STOP_MIN_DELTA
                print_with_time(
                    f'[验证 Episode {i_e}] cost={validation_cost:.2f}, '
                    f'best={best_validation_cost:.2f}, improved={improved}'
                )
                print_with_time(
                    f"[验证明细] fuel={metrics['fuel_cost']:.2f}, "
                    f"overtime={metrics['overtime']:.2f}, disc={metrics['discontinuity']:.2f}, "
                    f"station_ot={metrics['station_ot']:.2f}, project_ot={metrics['project_ot']:.2f}"
                )

                if improved:
                    best_validation_cost = validation_cost
                    best_validation_episode = i_e
                    no_improve_count = 0
                    save_validation_best(reposition_agent, i_e, metrics, validation_dates)
                    print_with_time(
                        f'验证集提升，保存best checkpoint: episode={i_e}, '
                        f'validation_cost={validation_cost:.2f}'
                    )
                else:
                    no_improve_count += 1
                    print_with_time(
                        f'验证集未提升，连续未提升次数={no_improve_count}'
                    )

                if EARLY_STOP_PATIENCE > 0 and no_improve_count >= EARLY_STOP_PATIENCE:
                    print_with_time(
                        f'触发早停：best_episode={best_validation_episode}, '
                        f'best_validation_cost={best_validation_cost:.2f}'
                    )
                    break

        print_with_time(f'\n---------------------------训练完成------------------------------')
        if validation_enabled and best_validation_episode >= 0:
            print_with_time(
                f'训练完成best checkpoint: episode={best_validation_episode}, '
                f'validation_cost={best_validation_cost:.2f}'
            )

    else:
        print_with_time('---------------------------测试模式------------------------------')
        use_dates = test_dates
        for test_idx in range(TEST_REPEAT_NUM):
            for day_idx, chosen_day in enumerate(use_dates):
                print_with_time(f'\n[测试 第{test_idx + 1}轮 第{day_idx + 1}天] 日期: {chosen_day}')
                run_one_day(env, chosen_day, dispatch_agent, reposition_agent, perception)

    env.print_result()


if __name__ == '__main__':
    main()
