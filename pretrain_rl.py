"""
预训练 RL Reposition 网络，使其行为逼近 UrgentRepositionAgent。
用 Urgent 的排序规则生成目标分数，回归训练 StationScoringNet，
然后将权重保存为 checkpoint_epoch0.pt，可供 test.ps1 直接加载。

Urgent 规则: argmin(truck_num, distance)
对应目标分数: score = -(w_truck * truck_num + w_dist * distance)
其中 w_truck >> w_dist 保证 truck_num 优先
"""

import numpy as np
import torch
from torch import nn
from component import *
from simulator import Environment
from parameter import *
from agent.reposition.urgent_agent import UrgentRepositionAgent
from agent.dispatch.fastest_agent import FastestDispatchAgent
from network.base_net import StationScoringNet
from toolkit.time import print_with_time
import datetime
import os


W_TRUCK = 10000.0   # truck_num 权重（主导）
W_DIST = 1.0        # distance 权重（次键 tiebreaker）


def compute_target_score(features_array: np.ndarray) -> np.ndarray:
    """给每个厂的 features 计算目标分数：= -(W_TRUCK * truck_num + W_DIST * dist)
    features: [dispatch_count, truck_num, dist, next_return, returning_count, future30, future60, future120]
    """
    truck_num = features_array[:, 1]
    dist = features_array[:, 2]
    return -(W_TRUCK * truck_num + W_DIST * dist)


def collect_training_data(env: Environment, dates: list) -> list[np.ndarray]:
    """运行模拟器，收集每次 reposition 时的厂站特征"""
    dispatch_agent = FastestDispatchAgent()
    reposition_agent = UrgentRepositionAgent(train_mode=False)
    all_features = []

    for day in dates:
        env.reset(day)

        while env.truck_iter or env.dispatch_iter or env.order_iter or env.unsolved_dispatch:

            if not env.truck_iter and not env.dispatch_iter and not env.order_iter and env.unsolved_dispatch:
                env.resolve_unsolved_dispatches()
                continue

            event_lt = env.truck_iter.next_event_lt()
            dispatch_lt = env.dispatch_iter.next_dispatch_lt(env.current_time)
            order_lt = env.order_iter.next_order_lt(env.current_time)

            lt_list = [event_lt, dispatch_lt, order_lt]
            index = np.argmin(lt_list)
            lt = lt_list[index]

            env.time_pass(lt)

            if index == 0:
                current_truck = env.truck_iter.return_next_event()
                assert current_truck.left_time == 0
                env.truck_state_change(current_truck)

                if current_truck.state == TruckState.Return:
                    env.finish_order_once(current_truck.oid)
                    current_pid = current_truck.to_pid
                    available_sids = list(env.interaction[current_pid].keys())

                    features_list = []
                    for sid in available_sids:
                        features = env.get_action_feature(sid, current_pid)
                        features_list.append(features)

                    all_features.append(np.array(features_list, dtype=np.float32))

                    chosen_sid = reposition_agent.select_reposition_station(
                        current_pid, current_truck, available_sids, env)
                    env.do_return(chosen_sid, current_truck)

                if not current_truck.state == TruckState.Free:
                    env.truck_iter.insert_in_order(current_truck)
                elif current_truck.state == TruckState.Free:
                    del current_truck

            elif index == 1:
                current_dispatch = env.dispatch_iter.return_next_dispatch()
                if current_dispatch.oid in env.fail_oid_set:
                    continue
                ret = env.execute_single_dispatch(current_dispatch)
                if not ret:
                    env.fail_dispatch += 1
                    env.unsolved_dispatch.append(current_dispatch)

            elif index == 2:
                current_order = env.order_iter.return_next_order()
                env.unfinished_orders[current_order.oid] = current_order
                env.not_fully_dispatch_orders[current_order.oid] = current_order
                available_sids = list(env.interaction[current_order.pid].keys())
                available_sids = [sid for sid in available_sids if sid in env.open_sid_set]

                for i in range(current_order.n_need):
                    select_sid = dispatch_agent.select_a_station(current_order, available_sids, env)
                    new_dispatch = Dispatch(current_order.oid, current_order.pid, select_sid, env.current_time)
                    new_dispatch.is_first_dispatch = (i == 0)
                    if current_order.oid in env.fail_oid_set:
                        continue
                    ret = env.execute_single_dispatch(new_dispatch)
                    if not ret:
                        env.fail_dispatch += 1
                        env.unsolved_dispatch.append(new_dispatch)

        env.record_site_overtime()

    return all_features


def compute_norm_stats(features_list: list, input_dim=8):
    all_f = np.vstack(features_list)
    mean = all_f.mean(axis=0).astype(np.float32)
    std = all_f.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def train_model(features_list: list, feat_mean, feat_std,
                input_dim=8, hidden_dims=(64, 32),
                epochs=100, batch_size=128, lr=0.001):
    """回归训练：网络输出分数逼近 compute_target_score"""
    device = DEVICE
    model = StationScoringNet(input_dim, hidden_dims, normalize=True).to(device)
    model.set_normalization(feat_mean, feat_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    # 为每个样本预计算 target（每个厂站的分数）
    targets = [compute_target_score(f).astype(np.float32) for f in features_list]

    split = int(len(features_list) * 0.9)
    train_f = features_list[:split]
    train_t = targets[:split]
    val_f = features_list[split:]
    val_t = targets[split:]

    print_with_time(f"训练集 {len(train_f)} 条, 验证集 {len(val_f)} 条, 共 {epochs} 轮")

    for epoch in range(epochs):
        indices = np.random.permutation(len(train_f))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(train_f), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_f = [train_f[i] for i in batch_idx]
            batch_t = [train_t[i] for i in batch_idx]

            max_s = max(f.shape[0] for f in batch_f)
            padded = np.zeros((len(batch_f), max_s, input_dim), dtype=np.float32)
            tgt = np.zeros((len(batch_f), max_s), dtype=np.float32)

            for i in range(len(batch_f)):
                n = batch_f[i].shape[0]
                padded[i, :n, :] = batch_f[i]
                tgt[i, :n] = batch_t[i]

            padded_t = torch.tensor(padded).to(device)
            tgt_t = torch.tensor(tgt).to(device)

            scores = model(padded_t).squeeze(-1)
            loss = criterion(scores, tgt_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            val_loss, val_acc = evaluate_model(model, val_f, val_t, device, input_dim)
            print_with_time(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / n_batches:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

    return model, optimizer


def evaluate_model(model, features_list, targets, device, input_dim):
    """评估：MSE loss + 准确率（argmax 选中的站是否与 Urgent 一致）"""
    model.eval()
    total_loss = 0
    correct = 0
    criterion = nn.MSELoss()

    for f, t in zip(features_list, targets):
        x = torch.tensor(f, dtype=torch.float).unsqueeze(0).to(device)
        t_t = torch.tensor(t, dtype=torch.float).unsqueeze(0).to(device)
        scores = model(x)
        total_loss += criterion(scores, t_t).item()

        pred = torch.argmax(scores).item()
        true = np.argmax(t)
        if pred == true:
            correct += 1

    model.train()
    return total_loss / len(features_list), correct / len(features_list)


def main():
    print_with_time("===== 预训练 RL Reposition 网络 =====")

    env = Environment()
    env.reset_statistic()

    all_dates = []
    current = EXPERIMENT_START_DATE.date()
    end = EXPERIMENT_END_DATE.date()
    while current <= end:
        all_dates.append(current)
        current += datetime.timedelta(days=1)

    split_idx = int(len(all_dates) * TRAIN_TEST_SPLIT_RATIO)
    train_dates = all_dates[:split_idx]

    print_with_time(f"在 {len(train_dates)} 个训练日上收集特征数据...")
    features_list = collect_training_data(env, train_dates)
    print_with_time(f"收集到 {len(features_list)} 条 reposition 样本")

    mean, std = compute_norm_stats(features_list)
    print_with_time(f"特征均值: {np.round(mean, 2)}, 标准差: {np.round(std, 2)}")

    model, optimizer = train_model(features_list, mean, std, epochs=100)
    val_acc = evaluate_model(model, features_list[-500:],
                             [compute_target_score(f) for f in features_list[-500:]],
                             DEVICE, 8)
    print_with_time(f"最终验证准确率: {val_acc[1]:.2%}")

    save_dir = MODEL_REPOSITION_RL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "checkpoint_epoch0_sim.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print_with_time(f"权重已保存到 {path}")
    print_with_time("现在可以用 run/DispatchAwareRL/test.ps1 加载该权重进行测试")


if __name__ == '__main__':
    main()
