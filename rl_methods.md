---

## 感知层（Perception Layer）

### 作用

感知层是 Dispatch 和 Reposition 之间的桥梁，记录每次决策的原意与实际结果，为 RL Reposition 提供"我的返程选择是否有利于后续调度"的反馈。

### 记录内容

**每次 Dispatch 记录：**

| 字段 | 含义 |
|------|------|
| `intent_sid` | Dispatch Agent 原始意图（理论上该选的厂站） |
| `actual_sid` | 实际选择的厂站（可能因无车而不同） |
| `station_truck_num` | 当时该厂站的可用车辆数 |
| `station_returned_count` | 该厂站累计收到多少辆返程车 |

**每次 Reposition 记录：**

| 字段 | 含义 |
|------|------|
| `from_pid` | 从哪个工地返回 |
| `chosen_sid` | 选择回到哪个厂站 |
| `received_positive` | 是否已获正奖励（防重复标记） |

### 奖励分配规则

**正奖励（+1）：** 当 Dispatch 意图与实际一致，且派出的车来源于之前的 Reposition 时，向最早一次回到该厂站的 Reposition 写入 +1。

**负奖励（−αᵗ）：** 当 Dispatch 意图与实际不一致时（想选 A 站但没车，选了 B 站），向该时刻之前的所有 Reposition 写入负奖励，按时间差衰减：`reward = -1 × 0.99ʰᵒᵘʳˢ`。

---

## 基础 RL Reposition（RLRepositionAgent）

### 状态

每个可用厂站的 8 维特征向量：

| 维度 | 特征 | 含义 |
|------|------|------|
| 0 | dispatch_count | 过去 1 小时该厂站发车数 |
| 1 | truck_num | 当前可用车辆数 |
| 2 | distance | 到工地的距离 (km) |
| 3 | next_return_time | 下一辆车预计返回时间 (分钟) |
| 4 | returning_count | 当前正在返回该厂站的车辆数 |
| 5 | future_30 | 未来 30 分钟计划发车数 |
| 6 | future_60 | 未来 60 分钟计划发车数 |
| 7 | future_120 | 未来 120 分钟计划发车数 |

### 动作

从当前工地的可选厂站列表中选择一个作为返程目标。

### 网络

两层 MLP（8→64→32→1），对每个厂站输出一个标量分数，通过 Softmax 得到概率分布。

### 奖励

两次 Reposition 之间的总成本增量 ÷ 时间间隔（取负），覆盖全部成本项：

| 成本项 | 说明 |
|--------|------|
| 装载油耗 | 重车去工地的油耗 × 油价 |
| 空返油耗 | 空车回厂站的油耗 × 油价 |
| 空转油耗 | 等待装载/浇筑时的油耗 × 油价 |
| 超时罚款 | 车辆到达晚于计划时间的罚款 |
| 连续浇筑罚款 | 两车间隔过大的罚款 |
| 加班补偿 | 厂站和工地的加班支出（均摊到所有 Reposition） |

最后一轮 Reposition 的奖励在当天结束时结算。

### 算法

REINFORCE（策略梯度）。每个 episode 结束后，从后向前累积折扣回报 G，计算梯度 `∇loss = -log π(a|s) × G`，统一更新网络参数。

---

## Dispatch 感知 RL（DispatchAwareRLRepositionAgent）

### 与基础 RL 的区别

在基础 RL 的全部逻辑之上，额外引入感知层的反馈信号。

### 双层奖励

1. **成本奖励**（可选，`use_cost_reward` 控制）：同基础 RL 的全部成本项
2. **感知层奖励**：每次 Dispatch 发生后，从感知层读取正负奖励，写入对应 Reposition 的 transition 中

当前默认 `use_cost_reward=True`，即 DispatchAwareRL 同时使用成本奖励和感知层奖励。仅使用感知层奖励时信号过稀疏，容易让训练出现不稳定或退化。

### 训练流程

```
dispatch 发生
  → 感知层记录 intent_vs_actual
  → process_dispatch_result() 计算正/负奖励
  → 写入对应 reposition 的 reward slot
```

RL 训练在 episode 结束后一次性执行，此时所有 reward slot 已被感知层和成本计算填满。

---

## 两种方法的适用场景

| | 基础 RL | Dispatch 感知 RL |
|---|---|---|
| 奖励来源 | 全部运行成本 | 感知层反馈（+可选成本） |
| 学习目标 | 最小化运营成本 | 最大化调度便利性 |
| 适用 | 独立优化 Reposition | 学习与 Dispatch 协同 |
| 预训练 | 可以（逼近 Urgent 等） | 可以（相同网络结构） |

---

## 使用

```powershell
# 从零预训练两套独立 Reposition 权重
conda run -n base python pretrain_reposition_policies.py

# 基础 RL 训练
.\run\RL\train.ps1

# Dispatch 感知 RL 训练
.\run\DispatchAwareRL\train.ps1

# 测试
.\run\RL\test.ps1
.\run\DispatchAwareRL\test.ps1

# 查看训练曲线
# 打开 check_learning_curve.ipynb
```

---

## 独立权重与重训练（2026-06-02改进）

原始实现中 RL 和 DispatchAwareRL 共用 `model/reposition/`，会导致两种方法加载同一套 checkpoint。现在权重分开存储：

| 方法 | 权重目录 |
|------|----------|
| RL | `model/reposition/rl/` |
| DispatchAwareRL | `model/reposition/dispatch_aware_rl/` |

旧的共享 checkpoint 已删除。两个方法分别从零重训：

1. `pretrain_reposition_policies.py` 在训练日期上收集两个不同需求缺口教师策略的数据，并分别训练两个网络。
2. RL 保留预训练后的 `model/reposition/rl/checkpoint_epoch0.pt`，因为短程策略梯度 fine-tune 后 epoch9 成本变差。
3. DispatchAwareRL 使用混合成本奖励和感知层奖励 fine-tune，保留 `model/reposition/dispatch_aware_rl/checkpoint_epoch9.pt`。

需求缺口教师策略使用的核心结构为：

```text
score =
  truck_num
  + 0.5 * inbound_count
  - 0.45 * (future_30 + 0.5 * future_60 + 0.25 * future_120)
  + 0.08 * distance
```

选择 `score` 最小的厂站。含义是：优先把车返回给当前车少、近期计划发车多、尚未有足够返程车补位、且返程距离不过远的厂站。参数集中在 `parameter.py`：

| 参数 | 默认值 |
|------|--------|
| `RL_USE_DEMAND_GAP_PRIOR` | `False` |
| `RL_DEMAND_GAP_DEMAND_WEIGHT` | `0.45` |
| `RL_DEMAND_GAP_DISTANCE_WEIGHT` | `0.08` |
| `RL_DEMAND_GAP_INBOUND_WEIGHT` | `0.5` |
| `RL_DEMAND_GAP_INBOUND_HORIZON` | `999` |
| `RL_DEMAND_GAP_FUTURE_30_WEIGHT` | `1.0` |
| `RL_DEMAND_GAP_FUTURE_60_WEIGHT` | `0.5` |
| `RL_DEMAND_GAP_FUTURE_120_WEIGHT` | `0.25` |

同一正式入口、同一 2024-05-01 至 2024-05-15 全量日期下的结果：

| Reposition 方法 | 总成本 RMB | 成本占收入 |
|-----------------|------------|------------|
| Urgent | `1,405,603.55` | `5.79%` |
| RL | `1,296,762.35` | `5.34%` |
| DispatchAwareRL | `1,287,579.69` | `5.30%` |

RL 比 Urgent 低 `108,841.20 RMB`；DispatchAwareRL 比 Urgent 低 `118,023.86 RMB`。两者加载不同目录下的不同 checkpoint，测试结果也不相同。REINFORCE 更新增加了回报标准化和梯度裁剪，降低继续训练时的退化风险。
