from agent.reposition.urgent_agent import UrgentRepositionAgent
from experiment_reposition_search import DemandGapRepositionAgent, PolicyConfig, evaluate_agent
from main_process import get_all_dates
from parameter import EXPERIMENT_END_DATE, EXPERIMENT_START_DATE


def main():
    dates = get_all_dates(EXPERIMENT_START_DATE, EXPERIMENT_END_DATE)
    urgent_metrics = evaluate_agent(UrgentRepositionAgent(train_mode=False), dates)
    urgent_cost = urgent_metrics["cost"]
    print(f"BASE Urgent all_cost={urgent_cost:.2f}", flush=True)

    configs: list[PolicyConfig] = []
    for demand_weight in (0.35, 0.45, 0.5, 0.6, 0.75):
        for distance_weight in (0.04, 0.05, 0.06, 0.08, 0.1):
            for inbound_weight in (0.5, 1.0, 1.5):
                configs.append(
                    PolicyConfig(
                        name=(
                            f"h999_dw{demand_weight}_dist{distance_weight}_"
                            f"in{inbound_weight}_rec0.0"
                        ),
                        horizon=999,
                        demand_weight=demand_weight,
                        distance_weight=distance_weight,
                        inbound_weight=inbound_weight,
                        recent_weight=0.0,
                    )
                )

    for demand_weight in (0.35, 0.45, 0.5, 0.6, 0.75):
        for inbound_weight in (0.5, 1.0, 1.5):
            configs.append(
                PolicyConfig(
                    name=f"lex_h999_dw{demand_weight}_in{inbound_weight}",
                    horizon=999,
                    demand_weight=demand_weight,
                    distance_weight=0.03,
                    inbound_weight=inbound_weight,
                    recent_weight=0.0,
                    lexicographic=True,
                )
            )

    best = []
    for idx, config in enumerate(configs, start=1):
        metrics = evaluate_agent(DemandGapRepositionAgent(config), dates)
        best.append((metrics["cost"], config, metrics))
        best.sort(key=lambda item: item[0])
        best = best[:10]
        if idx % 20 == 0:
            cost, cfg, _ = best[0]
            print(
                f"checked={idx}/{len(configs)} best={cost:.2f} "
                f"delta={cost - urgent_cost:.2f} {cfg.name}",
                flush=True,
            )

    print("TOP", flush=True)
    for cost, config, metrics in best:
        print(
            f"cost={cost:.2f} delta={cost - urgent_cost:.2f} name={config.name} "
            f"go={metrics['go_km']:.2f} return={metrics['return_km']:.2f} "
            f"overtime={metrics['overtime']:.2f} disc={metrics['discontinuity']:.2f} "
            f"station_ot={metrics['station_ot']:.2f} project_ot={metrics['project_ot']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
