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

每个可用厂站的 7 维特征向量：

| 维度 | 特征 | 含义 |
|------|------|------|
| 0 | dispatch_count | 过去 1 小时该厂站发车数 |
| 1 | truck_num | 当前可用车辆数 |
| 2 | distance | 到工地的距离 (km) |
| 3 | next_return_time | 下一辆车预计返回时间 (分钟) |
| 4 | future_30 | 未来 30 分钟计划发车数 |
| 5 | future_60 | 未来 60 分钟计划发车数 |
| 6 | future_120 | 未来 120 分钟计划发车数 |

### 动作

从当前工地的可选厂站列表中选择一个作为返程目标。

### 网络

两层 MLP（7→64→32→1），对每个厂站输出一个标量分数，通过 Softmax 得到概率分布。

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

`use_cost_reward=False`（默认）时仅使用感知层奖励，让网络纯粹学习"如何为后续调度提供便利"。

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
# 预训练（可选）
python pretrain_rl_synthetic.py

# 基础 RL 训练
.\run\RL\train.ps1

# Dispatch 感知 RL 训练
.\run\DispatchAwareRL\train.ps1

# 测试（自动加载 model/reposition/checkpoint_epoch*.pt）
.\run\RL\test.ps1
.\run\DispatchAwareRL\test.ps1

# 查看训练曲线
# 打开 check_learning_curve.ipynb
```
