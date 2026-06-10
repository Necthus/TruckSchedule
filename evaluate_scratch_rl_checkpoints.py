import os

from agent.dispatch.fastest_agent import FastestDispatchAgent
from agent.reposition.scratch_rl_agent import (
    ScratchDispatchAwareRLRepositionAgent,
    ScratchRLRepositionAgent,
)
from agent.reposition.urgent_agent import UrgentRepositionAgent
from main_process import get_all_dates, run_one_day, split_train_valid_test_dates
from parameter import (
    EMPTY_RETURN_FUEL_CONSUMPTION_PER_KM,
    EXPERIMENT_END_DATE,
    EXPERIMENT_START_DATE,
    LOAD_GO_FUEL_CONSUMPTION_PER_KM,
    MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_SAVE_DIR,
    MODEL_REPOSITION_SCRATCH_RL_SAVE_DIR,
    OIL_PRICE,
    TRAIN_TEST_SPLIT_RATIO,
    VALIDATION_SPLIT_RATIO,
)
from perception import PerceptionLayer
from simulator import Environment


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


def evaluate_agent(agent, dates):
    env = Environment()
    env.reset_statistic()
    dispatch_agent = FastestDispatchAgent()
    perception = PerceptionLayer()

    for day in dates:
        run_one_day(env, day, dispatch_agent, agent, perception)

    return {
        "cost": total_cost(env),
        "fuel_cost": (
            env.fuel_consumption
            + env.go_distance * LOAD_GO_FUEL_CONSUMPTION_PER_KM
            + env.return_distance * EMPTY_RETURN_FUEL_CONSUMPTION_PER_KM
        ) * OIL_PRICE,
        "overtime": env.overtime_penalty,
        "discontinuity": env.discontinuity_penalty,
        "station_ot": env.station_overtime_cost,
        "project_ot": env.project_overtime_cost,
        "go_km": env.go_distance,
        "return_km": env.return_distance,
    }


def list_epochs(save_dir):
    epochs = []
    if not os.path.exists(save_dir):
        return epochs
    for name in os.listdir(save_dir):
        if name.startswith("checkpoint_epoch") and name.endswith(".pt"):
            epochs.append(int(name[len("checkpoint_epoch"):-len(".pt")]))
    return sorted(epochs)


def evaluate_checkpoint(name, agent_cls, save_dir, epoch, dates, urgent_cost, split_name):
    agent = agent_cls(train_mode=False)
    agent.algo.load(save_dir, epoch)
    metrics = evaluate_agent(agent, dates)
    print(
        f"{split_name} {name} epoch={epoch} cost={metrics['cost']:.2f} "
        f"delta_vs_urgent={metrics['cost'] - urgent_cost:.2f} "
        f"fuel={metrics['fuel_cost']:.2f} overtime={metrics['overtime']:.2f} "
        f"disc={metrics['discontinuity']:.2f} station_ot={metrics['station_ot']:.2f} "
        f"project_ot={metrics['project_ot']:.2f}",
        flush=True,
    )
    return metrics


def evaluate_checkpoints(name, agent_cls, save_dir, dates, urgent_cost, split_name):
    rows = []
    for epoch in list_epochs(save_dir):
        metrics = evaluate_checkpoint(name, agent_cls, save_dir, epoch, dates, urgent_cost, split_name)
        rows.append((metrics["cost"], epoch, metrics))

    rows.sort(key=lambda item: item[0])
    if rows:
        best_cost, best_epoch, _ = rows[0]
        print(
            f"BEST {split_name} {name} epoch={best_epoch} cost={best_cost:.2f} "
            f"delta_vs_urgent={best_cost - urgent_cost:.2f}",
            flush=True,
        )
        return best_epoch, best_cost
    return None, None


def main():
    print(
        "AUDIT ONLY: training now selects checkpoints during periodic validation. "
        "Do not use this script to choose production/test checkpoints.",
        flush=True,
    )
    all_dates = get_all_dates(EXPERIMENT_START_DATE, EXPERIMENT_END_DATE)
    train_dates, validation_dates, test_dates = split_train_valid_test_dates(
        all_dates,
        TRAIN_TEST_SPLIT_RATIO,
        VALIDATION_SPLIT_RATIO,
    )

    print(
        f"dates train={len(train_dates)} {train_dates[0]}..{train_dates[-1]} "
        f"validation={len(validation_dates)} {validation_dates[0]}..{validation_dates[-1]} "
        f"test={len(test_dates)} {test_dates[0]}..{test_dates[-1]}",
        flush=True,
    )

    urgent_validation = evaluate_agent(UrgentRepositionAgent(train_mode=False), validation_dates)
    urgent_validation_cost = urgent_validation["cost"]
    print(f"VALIDATION Urgent cost={urgent_validation_cost:.2f}", flush=True)

    scratch_best_epoch, _ = evaluate_checkpoints(
        "ScratchRL",
        ScratchRLRepositionAgent,
        MODEL_REPOSITION_SCRATCH_RL_SAVE_DIR,
        validation_dates,
        urgent_validation_cost,
        "VALIDATION",
    )
    scratch_dispatch_best_epoch, _ = evaluate_checkpoints(
        "ScratchDispatchAwareRL",
        ScratchDispatchAwareRLRepositionAgent,
        MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_SAVE_DIR,
        validation_dates,
        urgent_validation_cost,
        "VALIDATION",
    )

    urgent_test = evaluate_agent(UrgentRepositionAgent(train_mode=False), test_dates)
    urgent_test_cost = urgent_test["cost"]
    print(f"TEST Urgent cost={urgent_test_cost:.2f}", flush=True)

    if scratch_best_epoch is not None:
        evaluate_checkpoint(
            "ScratchRL",
            ScratchRLRepositionAgent,
            MODEL_REPOSITION_SCRATCH_RL_SAVE_DIR,
            scratch_best_epoch,
            test_dates,
            urgent_test_cost,
            "TEST",
        )
        print(f"AUDIT BEST MODEL_REPOSITION_SCRATCH_RL_BEST_EPOCH = {scratch_best_epoch}", flush=True)

    if scratch_dispatch_best_epoch is not None:
        evaluate_checkpoint(
            "ScratchDispatchAwareRL",
            ScratchDispatchAwareRLRepositionAgent,
            MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_SAVE_DIR,
            scratch_dispatch_best_epoch,
            test_dates,
            urgent_test_cost,
            "TEST",
        )
        print(
            "AUDIT BEST MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_BEST_EPOCH = "
            f"{scratch_dispatch_best_epoch}",
            flush=True,
        )


if __name__ == "__main__":
    main()
