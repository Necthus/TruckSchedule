from dataclasses import dataclass

from parameter import (
    RL_DEMAND_GAP_DEMAND_WEIGHT,
    RL_DEMAND_GAP_DISTANCE_WEIGHT,
    RL_DEMAND_GAP_FUTURE_120_WEIGHT,
    RL_DEMAND_GAP_FUTURE_30_WEIGHT,
    RL_DEMAND_GAP_FUTURE_60_WEIGHT,
    RL_DEMAND_GAP_INBOUND_HORIZON,
    RL_DEMAND_GAP_INBOUND_WEIGHT,
)


@dataclass(frozen=True)
class DemandGapConfig:
    demand_weight: float = RL_DEMAND_GAP_DEMAND_WEIGHT
    distance_weight: float = RL_DEMAND_GAP_DISTANCE_WEIGHT
    inbound_weight: float = RL_DEMAND_GAP_INBOUND_WEIGHT
    inbound_horizon: float = RL_DEMAND_GAP_INBOUND_HORIZON
    future_30_weight: float = RL_DEMAND_GAP_FUTURE_30_WEIGHT
    future_60_weight: float = RL_DEMAND_GAP_FUTURE_60_WEIGHT
    future_120_weight: float = RL_DEMAND_GAP_FUTURE_120_WEIGHT


DEFAULT_DEMAND_GAP_CONFIG = DemandGapConfig()


def demand_gap_score(current_pid, sid, env, config: DemandGapConfig = DEFAULT_DEMAND_GAP_CONFIG):
    """Lower score means the station has a larger near-future truck gap."""
    station = env.stations[sid]
    _dispatch_count, truck_num, distance, _next_return, returning_count, future_30, future_60, future_120 = (
        env.get_action_feature(sid, current_pid)
    )
    future_need = (
        config.future_30_weight * future_30
        + config.future_60_weight * future_60
        + config.future_120_weight * future_120
    )
    if config.inbound_horizon >= 999:
        inbound_count = returning_count
    else:
        inbound_count = sum(
            1 for return_time in station.return_time_list
            if return_time <= config.inbound_horizon
        )
    return (
        truck_num
        + config.inbound_weight * inbound_count
        - config.demand_weight * future_need
        + config.distance_weight * distance
    )


def select_demand_gap_action(current_pid, available_sids, env, config: DemandGapConfig = DEFAULT_DEMAND_GAP_CONFIG):
    action_idx = min(
        range(len(available_sids)),
        key=lambda idx: demand_gap_score(current_pid, available_sids[idx], env, config),
    )
    return action_idx
