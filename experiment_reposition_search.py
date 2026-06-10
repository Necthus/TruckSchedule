import datetime
from dataclasses import dataclass

from agent.dispatch.fastest_agent import FastestDispatchAgent
from agent.reposition.base_agent import BaseRepositionAgent
from agent.reposition.urgent_agent import UrgentRepositionAgent
from component import Truck
from main_process import get_all_dates, run_one_day
from parameter import (
    EMPTY_RETURN_FUEL_CONSUMPTION_PER_KM,
    EXPERIMENT_END_DATE,
    EXPERIMENT_START_DATE,
    LOAD_GO_FUEL_CONSUMPTION_PER_KM,
    OIL_PRICE,
)
from perception import PerceptionLayer
from simulator import Environment


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    horizon: int
    demand_weight: float
    distance_weight: float
    inbound_weight: float
    recent_weight: float
    lexicographic: bool = False


class DemandGapRepositionAgent(BaseRepositionAgent):
    """Deterministic policy used to search for an RL prior better than Urgent."""

    def __init__(self, config: PolicyConfig):
        super().__init__(train_mode=False)
        self.config = config

    def select_reposition_station(self, current_pid, truck: Truck, available_sids, env: Environment):
        cfg = self.config

        def station_score(sid):
            s = env.stations[sid]
            dispatch_count, truck_num, dist, _next_return, returning_count, fut30, fut60, fut120 = env.get_action_feature(
                sid, current_pid
            )
            if cfg.horizon == 30:
                future_need = fut30
            elif cfg.horizon == 60:
                future_need = fut60
            elif cfg.horizon == 120:
                future_need = fut120
            else:
                future_need = fut30 + 0.5 * fut60 + 0.25 * fut120

            if cfg.horizon >= 999:
                inbound = returning_count
            else:
                inbound = sum(1 for t in s.return_time_list if t <= cfg.horizon)
            current_gap = truck_num + cfg.inbound_weight * inbound - cfg.demand_weight * future_need
            recent_gap = current_gap - cfg.recent_weight * dispatch_count

            if cfg.lexicographic:
                return (recent_gap, cfg.distance_weight * dist)

            return recent_gap + cfg.distance_weight * dist

        return min(available_sids, key=station_score)


def compute_total_cost(env: Environment) -> float:
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


def evaluate_agent(agent: BaseRepositionAgent, dates: list[datetime.date]) -> dict:
    env = Environment()
    env.reset_statistic()
    dispatch_agent = FastestDispatchAgent()
    perception = PerceptionLayer()

    for day in dates:
        run_one_day(env, day, dispatch_agent, agent, perception)

    return {
        "cost": compute_total_cost(env),
        "go_km": env.go_distance,
        "return_km": env.return_distance,
        "overtime": env.overtime_penalty,
        "discontinuity": env.discontinuity_penalty,
        "station_ot": env.station_overtime_cost,
        "project_ot": env.project_overtime_cost,
    }


def main():
    dates = get_all_dates(EXPERIMENT_START_DATE, EXPERIMENT_END_DATE)
    split = int(len(dates) * 0.8)
    train_dates = dates[:split]
    urgent_metrics = evaluate_agent(UrgentRepositionAgent(train_mode=False), dates)
    urgent_train_metrics = evaluate_agent(UrgentRepositionAgent(train_mode=False), train_dates)
    print(
        f"BASE Urgent train_cost={urgent_train_metrics['cost']:.2f} "
        f"all_cost={urgent_metrics['cost']:.2f}",
        flush=True,
    )

    configs: list[PolicyConfig] = []
    for horizon in (60, 120, 999):
        for demand_weight in (0.5, 1.0, 2.0):
            for distance_weight in (0.0, 0.03, 0.05):
                for inbound_weight in (0.0, 1.0):
                    for recent_weight in (0.0,):
                        name = (
                            f"h{horizon}_dw{demand_weight}_dist{distance_weight}_"
                            f"in{inbound_weight}_rec{recent_weight}"
                        )
                        configs.append(
                            PolicyConfig(
                                name=name,
                                horizon=horizon,
                                demand_weight=demand_weight,
                                distance_weight=distance_weight,
                                inbound_weight=inbound_weight,
                                recent_weight=recent_weight,
                            )
                        )
    for horizon in (60, 120, 999):
        for demand_weight in (0.5, 1.0, 2.0):
            for distance_weight in (0.03, 0.05):
                name = f"lex_h{horizon}_dw{demand_weight}_dist{distance_weight}"
                configs.append(
                    PolicyConfig(
                        name=name,
                        horizon=horizon,
                        demand_weight=demand_weight,
                        distance_weight=distance_weight,
                        inbound_weight=1.0,
                        recent_weight=0.0,
                        lexicographic=True,
                    )
                )

    train_best = []
    for idx, config in enumerate(configs, start=1):
        metrics = evaluate_agent(DemandGapRepositionAgent(config), train_dates)
        train_best.append((metrics["cost"], config, metrics))
        train_best.sort(key=lambda x: x[0])
        train_best = train_best[:20]
        if idx % 10 == 0:
            current = train_best[0]
            print(
                f"checked={idx}/{len(configs)} train_best={current[0]:.2f} "
                f"delta={current[0] - urgent_train_metrics['cost']:.2f} {current[1].name}",
                flush=True,
            )

    best = []
    for _, config, _ in train_best:
        metrics = evaluate_agent(DemandGapRepositionAgent(config), dates)
        best.append((metrics["cost"], config, metrics))
    best.sort(key=lambda x: x[0])

    print("TOP", flush=True)
    for cost, config, metrics in best[:10]:
        print(
            f"cost={cost:.2f} delta={cost - urgent_metrics['cost']:.2f} "
            f"name={config.name} return={metrics['return_km']:.2f} "
            f"disc={metrics['discontinuity']:.2f} station_ot={metrics['station_ot']:.2f} "
            f"project_ot={metrics['project_ot']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
