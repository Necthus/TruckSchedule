import os

import numpy as np
import torch
from torch import nn

from agent.dispatch.fastest_agent import FastestDispatchAgent
from agent.reposition.base_agent import BaseRepositionAgent
from agent.reposition.demand_gap_policy import DemandGapConfig, select_demand_gap_action
from component import Truck
from main_process import get_all_dates, run_one_day
from network.base_net import StationScoringNet
from parameter import (
    DEVICE,
    EXPERIMENT_END_DATE,
    EXPERIMENT_START_DATE,
    MODEL_REPOSITION_DISPATCH_AWARE_SAVE_DIR,
    MODEL_REPOSITION_RL_SAVE_DIR,
    SEED,
    TRAIN_TEST_SPLIT_RATIO,
)
from perception import PerceptionLayer
from simulator import Environment
from toolkit.time import print_with_time


INPUT_DIM = 8
HIDDEN_DIMS = (64, 32)
EPOCHS = 300
BATCH_SIZE = 128
LR = 0.001


RL_TEACHER_CONFIG = DemandGapConfig(
    demand_weight=0.45,
    distance_weight=0.08,
    inbound_weight=0.5,
    inbound_horizon=999,
    future_30_weight=1.0,
    future_60_weight=0.5,
    future_120_weight=0.25,
)

DISPATCH_AWARE_TEACHER_CONFIG = DemandGapConfig(
    demand_weight=0.35,
    distance_weight=0.06,
    inbound_weight=0.5,
    inbound_horizon=999,
    future_30_weight=1.0,
    future_60_weight=0.5,
    future_120_weight=0.25,
)


class RecordingDemandGapAgent(BaseRepositionAgent):
    def __init__(self, config: DemandGapConfig):
        super().__init__(train_mode=False)
        self.config = config
        self.states = []
        self.actions = []

    def select_reposition_station(self, current_pid, truck: Truck, available_sids, env: Environment):
        features = np.array(
            [env.get_action_feature(sid, current_pid) for sid in available_sids],
            dtype=np.float32,
        )
        action_idx = select_demand_gap_action(current_pid, available_sids, env, self.config)
        self.states.append(features)
        self.actions.append(action_idx)
        return available_sids[action_idx]


def collect_teacher_data(config: DemandGapConfig, dates):
    env = Environment()
    env.reset_statistic()
    dispatch_agent = FastestDispatchAgent()
    reposition_agent = RecordingDemandGapAgent(config)
    perception = PerceptionLayer()

    for day in dates:
        run_one_day(env, day, dispatch_agent, reposition_agent, perception)

    return reposition_agent.states, reposition_agent.actions


def compute_norm_stats(states):
    stacked = np.vstack(states)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def train_policy(states, actions):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mean, std = compute_norm_stats(states)
    model = StationScoringNet(INPUT_DIM, HIDDEN_DIMS, normalize=True).to(DEVICE)
    model.set_normalization(mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    n = len(states)
    for epoch in range(EPOCHS):
        indices = np.random.permutation(n)
        total_loss = 0.0
        batch_count = 0

        for start in range(0, n, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            batch_states = [states[i] for i in batch_idx]
            batch_actions = [actions[i] for i in batch_idx]

            max_actions = max(state.shape[0] for state in batch_states)
            padded = np.zeros((len(batch_states), max_actions, INPUT_DIM), dtype=np.float32)
            mask = np.ones((len(batch_states), max_actions), dtype=bool)

            for row, state in enumerate(batch_states):
                action_count = state.shape[0]
                padded[row, :action_count, :] = state
                mask[row, :action_count] = False

            x = torch.tensor(padded, dtype=torch.float32).to(DEVICE)
            mask_tensor = torch.tensor(mask, dtype=torch.bool).to(DEVICE)
            y = torch.tensor(batch_actions, dtype=torch.long).to(DEVICE)

            scores = model(x)
            scores = scores.masked_fill(mask_tensor, -1e9)
            loss = criterion(scores, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if (epoch + 1) % 50 == 0:
            acc = evaluate_accuracy(model, states, actions)
            print_with_time(
                f"epoch={epoch + 1}/{EPOCHS} loss={total_loss / batch_count:.4f} acc={acc:.2%}"
            )

    return model, optimizer


def evaluate_accuracy(model, states, actions):
    model.eval()
    correct = 0
    with torch.no_grad():
        for state, action in zip(states, actions):
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = torch.argmax(model(x).squeeze(0)).item()
            if pred == action:
                correct += 1
    model.train()
    return correct / len(states)


def save_checkpoint(model, optimizer, save_dir, config: DemandGapConfig):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "checkpoint_epoch0.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "teacher_config": config.__dict__,
            "pretrain": "demand_gap_cross_entropy",
        },
        path,
    )
    return path


def pretrain_one(name, config, save_dir, dates):
    print_with_time(f"collect {name} teacher data")
    states, actions = collect_teacher_data(config, dates)
    print_with_time(f"{name} samples={len(states)} save_dir={save_dir}")
    model, optimizer = train_policy(states, actions)
    acc = evaluate_accuracy(model, states, actions)
    path = save_checkpoint(model, optimizer, save_dir, config)
    print_with_time(f"{name} final_acc={acc:.2%} saved={path}")


def main():
    all_dates = get_all_dates(EXPERIMENT_START_DATE, EXPERIMENT_END_DATE)
    split_idx = int(len(all_dates) * TRAIN_TEST_SPLIT_RATIO)
    train_dates = all_dates[:split_idx]
    print_with_time(f"pretrain dates={train_dates[0]}..{train_dates[-1]} count={len(train_dates)}")
    pretrain_one("RL", RL_TEACHER_CONFIG, MODEL_REPOSITION_RL_SAVE_DIR, train_dates)
    pretrain_one(
        "DispatchAwareRL",
        DISPATCH_AWARE_TEACHER_CONFIG,
        MODEL_REPOSITION_DISPATCH_AWARE_SAVE_DIR,
        train_dates,
    )


if __name__ == "__main__":
    main()
