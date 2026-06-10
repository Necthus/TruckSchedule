"""
纯人造数据预训练 — 不需要模拟器。
直接生成厂站特征，用 Urgent 规则打目标分数，MSE 回归训练。
truck_num 越低分数越高，distance 做 tiebreaker。
"""

import numpy as np
import torch
from torch import nn
from network.base_net import StationScoringNet
from parameter import DEVICE, MODEL_REPOSITION_RL_SAVE_DIR
from toolkit.time import print_with_time
import os

# ── 参数 ──────────────────────────────────────────────
NUM_SAMPLES = 20000        # 人造样本数
MAX_STATIONS = 20          # 每次最多可选厂站数
INPUT_DIM = 8
HIDDEN_DIMS = (64, 32)
EPOCHS = 500
BATCH_SIZE = 256
LR = 0.001

# 目标分数: score = -(W1 * truck_num + W2 * distance)
W1 = 10000.0   # truck_num 主导
W2 = 1.0       # distance tiebreaker


def generate_synthetic_data(n=NUM_SAMPLES, max_stations=MAX_STATIONS):
    """生成人造数据，特征范围贴近真实数据分布"""
    all_features = []
    all_targets = []

    for _ in range(n):
        num_stations = np.random.randint(5, max_stations + 1)

        # 模拟真实特征分布（参考从实际数据算出的均值/标准差）
        truck_num = np.random.randint(0, 16, size=num_stations).astype(np.float32)
        distance = np.random.uniform(1, 30, size=num_stations).astype(np.float32)
        dispatch_count = np.random.poisson(2, size=num_stations).astype(np.float32)
        next_return = np.random.uniform(0, 240, size=num_stations).astype(np.float32)
        returning_count = np.random.poisson(1, size=num_stations).astype(np.float32)
        future_30 = np.random.poisson(0.5, size=num_stations).astype(np.float32)
        future_60 = np.random.poisson(1, size=num_stations).astype(np.float32)
        future_120 = np.random.poisson(2, size=num_stations).astype(np.float32)

        features = np.stack([
            dispatch_count, truck_num, distance,
            next_return, returning_count, future_30, future_60, future_120
        ], axis=1)  # (num_stations, 8)

        # 目标分数 = -(W1 * truck_num + W2 * distance)
        targets = -(W1 * truck_num + W2 * distance)

        all_features.append(features)
        all_targets.append(targets)

    return all_features, all_targets


def compute_norm_stats(features_list):
    all_f = np.vstack(features_list)
    mean = all_f.mean(axis=0).astype(np.float32)
    std = all_f.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def train(features_list, targets_list, feat_mean, feat_std):
    device = DEVICE
    model = StationScoringNet(INPUT_DIM, HIDDEN_DIMS, normalize=True).to(device)
    model.set_normalization(feat_mean, feat_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    criterion = nn.MSELoss()

    n = len(features_list)
    print_with_time(f"人造数据 {n} 条, 训练 {EPOCHS} 轮, batch_size={BATCH_SIZE}")

    for epoch in range(EPOCHS):
        indices = np.random.permutation(n)
        total_loss = 0
        n_batches = 0

        for start in range(0, n, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            batch_f = [features_list[i] for i in batch_idx]
            batch_t = [targets_list[i] for i in batch_idx]

            max_s = max(f.shape[0] for f in batch_f)
            padded = np.zeros((len(batch_f), max_s, INPUT_DIM), dtype=np.float32)
            tgt = np.zeros((len(batch_f), max_s), dtype=np.float32)

            for i in range(len(batch_f)):
                m = batch_f[i].shape[0]
                padded[i, :m, :] = batch_f[i]
                tgt[i, :m] = batch_t[i]

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
            print_with_time(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / n_batches:.2f}")

    return model, optimizer


def evaluate(model, features_list, targets_list):
    model.eval()
    correct = 0
    for f, t in zip(features_list, targets_list):
        x = torch.tensor(f, dtype=torch.float).unsqueeze(0).to(DEVICE)
        scores = model(x).squeeze(0)
        pred = torch.argmax(scores).item()
        true = np.argmax(t).item()
        if pred == true:
            correct += 1
    model.train()
    return correct / len(features_list)


def main():
    print_with_time("===== 人造数据预训练 =====")

    features_list, targets_list = generate_synthetic_data()
    mean, std = compute_norm_stats(features_list)
    print_with_time(f"归一化 mean={np.round(mean, 2)}, std={np.round(std, 2)}")

    model, optimizer = train(features_list, targets_list, mean, std)

    acc = evaluate(model, features_list[-1000:], targets_list[-1000:])
    print_with_time(f"准确率: {acc:.2%}")

    os.makedirs(MODEL_REPOSITION_RL_SAVE_DIR, exist_ok=True)
    path = os.path.join(MODEL_REPOSITION_RL_SAVE_DIR, "checkpoint_epoch0.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print_with_time(f"权重已保存到 {path}")


if __name__ == '__main__':
    main()
