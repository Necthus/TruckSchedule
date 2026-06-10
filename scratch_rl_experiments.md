# 从零训练 Reposition RL 实验记录

本文档记录本轮不使用教师策略、不使用监督预训练权重的 Reposition RL 改进实验。

结论先写明：上一轮 `RL` / `DispatchAwareRL` 的主要提升确实来自需求缺口教师策略的监督预训练，不应称为“从零训练出的 RL”。本轮新增的 `ScratchRL` / `ScratchDispatchAwareRL` 使用随机初始化权重和策略梯度训练，权重目录独立，未加载上一轮教师预训练 checkpoint。

最新 8:1:1 shuffle 训练结果是：`ScratchRL` 使用训练中周期验证选择 epoch99 后，在测试集比 `Urgent` 低 `27,139.34 RMB`；`ScratchDispatchAwareRL` 使用训练中周期验证选择 epoch99 后，在测试集比 `Urgent` 低 `70,619.24 RMB`。本轮新增的 `ScratchCombinedRL` 使用独立权重目录，把 `ScratchRL` 的成本/shaping 奖励和感知层奖励融合，按训练前 20 episode 奖励量级校准出感知奖励系数 `1.1`；它使用训练中周期验证选择 epoch199 后，测试集比 `Urgent` 低 `67,795.09 RMB`，比 `ScratchRL` 低 `40,655.76 RMB`，但比当前最好的 `ScratchDispatchAwareRL` 高 `2,824.15 RMB`。

最新消融结论：`ScratchDispatchAwareRL` 不是纯 aware reward，而是 `cost + shaping + perception`。先前两个独立消融方法里，去掉 cost reward 的 `ScratchDispatchAwareNoCostRL` 测试成本为 `993,700.60 RMB`，比 `Urgent` 高 `6,262.05 RMB`；去掉 shaping reward 的 `ScratchDispatchAwareNoShapingRL` 测试成本为 `989,989.44 RMB`，比 `Urgent` 高 `2,550.89 RMB`。本轮继续新增纯信号消融：`ScratchCostOnlyRL` 只保留成本 reward，原 test split 成本 `967,388.12 RMB`，比 `Urgent` 低 `20,050.43 RMB`；`ScratchPerceptionOnlyRL` 只保留 perception reward，原 test split 成本 `1,009,744.50 RMB`，比 `Urgent` 高 `22,305.95 RMB`。因此当前原 test split 上最佳仍是 `ScratchDispatchAwareRL`，三信号组合最好；单独 cost 能超过 Urgent 但不如 `cost + shaping`，单独 perception 在原 test split 不稳。旧的顺序训练权重和指标已归档到 `archived_before_shuffle_8_1_1_metrics.md` 和 `model/reposition/archive_before_shuffle_8_1_1/`。

## 最新 8:1:1 Shuffle 实验

本轮修复了测试口径 bug：测试模式不再遍历 `EXPERIMENT_START_DATE ~ EXPERIMENT_END_DATE` 全量日期，而是只使用 test split。

默认日期范围已扩大为：

```text
EXPERIMENT_START_DATE = 2024-01-12
EXPERIMENT_END_DATE = 2024-06-04
TRAIN_TEST_SPLIT_RATIO = 0.8
VALIDATION_SPLIT_RATIO = 0.1
SHUFFLE_TRAIN_DATES = True
```

时间顺序切分：

| split | 日期数 | 日期范围 | 用途 |
|-------|--------|----------|------|
| train | 116 | `2024-01-12 ~ 2024-05-06` | 训练 policy，默认按 cycle 打乱 |
| validation | 14 | `2024-05-07 ~ 2024-05-20` | 选择 checkpoint |
| test | 15 | `2024-05-21 ~ 2024-06-04` | 最终测试，不参与选模型 |

训练命令：

```powershell
.\run\ScratchRL\train.ps1
.\run\ScratchDispatchAwareRL\train.ps1
.\run\ScratchDispatchAwareNoCostRL\train.ps1
.\run\ScratchDispatchAwareNoShapingRL\train.ps1
.\run\ScratchCombinedRL\train.ps1
.\run\ScratchCostOnlyRL\train.ps1
.\run\ScratchPerceptionOnlyRL\train.ps1
```

训练中验证流程：

1. 每 `VALIDATION_FREQUENCY=20` 个 episode，临时切到贪心测试模式，在 validation split 上评估当前策略。
2. 如果 validation cost 降低超过 `EARLY_STOP_MIN_DELTA`，立即保存当前 checkpoint，并写入 `best_checkpoint.json`。
3. 如果连续 `EARLY_STOP_PATIENCE=5` 次验证没有提升，则停止训练。
4. 测试模式优先读取 `best_checkpoint.json`，不再依赖训练结束后逐一扫描 checkpoint。
5. 训练日期每个 cycle 用 `SEED + cycle` 做 deterministic shuffle；validation/test 不打乱。

### 最新验证集选择

| 方法 | validation 最优 epoch | validation 成本 RMB | 相对 validation Urgent |
|------|-----------------------|---------------------|------------------------|
| `Urgent` | 无 | `1,520,301.63` | baseline |
| `ScratchRL` | `99` | `1,268,620.47` | `-251,681.17` |
| `ScratchDispatchAwareRL` | `99` | `1,314,243.65` | `-206,057.98` |
| `ScratchDispatchAwareNoCostRL` | `99` | `1,335,144.05` | `-185,157.58` |
| `ScratchDispatchAwareNoShapingRL` | `199` | `1,406,924.09` | `-113,377.54` |
| `ScratchCombinedRL` | `199` | `1,239,681.09` | `-280,620.54` |
| `ScratchCostOnlyRL` | `19` | `1,443,356.87` | `-76,944.76` |
| `ScratchPerceptionOnlyRL` | `19` | `1,355,021.16` | `-165,280.47` |

### 最新测试集结果

正式入口：

```powershell
python .\main_process.py --reposition_method Urgent
.\run\ScratchRL\test.ps1
.\run\ScratchDispatchAwareRL\test.ps1
.\run\ScratchDispatchAwareNoCostRL\test.ps1
.\run\ScratchDispatchAwareNoShapingRL\test.ps1
.\run\ScratchCombinedRL\test.ps1
.\run\ScratchCostOnlyRL\test.ps1
.\run\ScratchPerceptionOnlyRL\test.ps1
```

测试集只包含 `2024-05-21 ~ 2024-06-04`。

| 方法 | 默认加载 | test 成本 RMB | 相对 test Urgent |
|------|----------|---------------|------------------|
| `Urgent` | 无 | `987,438.55` | baseline |
| `ScratchRL` | `best_checkpoint.json -> checkpoint_epoch99.pt` | `960,299.22` | `-27,139.34` |
| `ScratchDispatchAwareRL` | `best_checkpoint.json -> checkpoint_epoch99.pt` | `916,819.31` | `-70,619.24` |
| `ScratchDispatchAwareNoCostRL` | `best_checkpoint.json -> checkpoint_epoch99.pt` | `993,700.60` | `+6,262.05` |
| `ScratchDispatchAwareNoShapingRL` | `best_checkpoint.json -> checkpoint_epoch199.pt` | `989,989.44` | `+2,550.89` |
| `ScratchCombinedRL` | `best_checkpoint.json -> checkpoint_epoch199.pt` | `919,643.46` | `-67,795.09` |
| `ScratchCostOnlyRL` | `best_checkpoint.json -> checkpoint_epoch19.pt` | `967,388.12` | `-20,050.43` |
| `ScratchPerceptionOnlyRL` | `best_checkpoint.json -> checkpoint_epoch19.pt` | `1,009,744.50` | `+22,305.95` |

当前默认配置：

```python
MODEL_REPOSITION_SCRATCH_RL_BEST_EPOCH = 99
MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_BEST_EPOCH = 99
MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_COST_BEST_EPOCH = 99
MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_SHAPING_BEST_EPOCH = 199
MODEL_REPOSITION_SCRATCH_COMBINED_BEST_EPOCH = 199
MODEL_REPOSITION_SCRATCH_COST_ONLY_BEST_EPOCH = 99
MODEL_REPOSITION_SCRATCH_PERCEPTION_ONLY_BEST_EPOCH = 99
```

上面的 fallback epoch 只在 `best_checkpoint.json` 不存在时使用；当前测试实际读取 metadata。与顺序训练相比，增加训练集长度并默认打乱训练后，`ScratchRL`、`ScratchDispatchAwareRL`、`ScratchCombinedRL`、`ScratchCostOnlyRL` 在 test split 上超过 Urgent；`ScratchDispatchAwareNoCostRL`、`ScratchDispatchAwareNoShapingRL`、`ScratchPerceptionOnlyRL` 没有超过 Urgent。当前最低测试成本仍是 `ScratchDispatchAwareRL`。

### Test Set2：2023-11 历史窗口

为了检查模型在训练期之前历史数据上的泛化，额外抽取 `2023-11-01 ~ 2023-11-15` 作为 test set2。该窗口约 15 天，模拟器当前时间过滤口径下包含 `1,002` 条订单、`6,005` 次运输需求、`55,511.50 m3` 方量。

评估脚本：

```powershell
python .\evaluate_test_set2_november.py
```

该脚本只做评估，不训练、不保存 checkpoint；RL 方法均读取各自已有 `best_checkpoint.json`。

| 方法 | reward 信号 | 加载 checkpoint | test set2 成本 RMB | 相对 Urgent |
|------|-------------|-----------------|--------------------|-------------|
| `ScratchPerceptionOnlyRL` | perception | `checkpoint_epoch19.pt` | `569,223.05` | `-38,130.21` |
| `ScratchDispatchAwareNoShapingRL` | cost + perception | `checkpoint_epoch199.pt` | `576,630.26` | `-30,723.00` |
| `ScratchRL` | cost + shaping | `checkpoint_epoch99.pt` | `578,443.40` | `-28,909.86` |
| `ScratchDispatchAwareRL` | cost + shaping + perception | `checkpoint_epoch99.pt` | `591,277.15` | `-16,076.11` |
| `ScratchCostOnlyRL` | cost | `checkpoint_epoch19.pt` | `606,904.28` | `-448.98` |
| `Urgent` | heuristic baseline | 无 | `607,353.26` | baseline |
| `ScratchDispatchAwareNoCostRL` | shaping + perception | `checkpoint_epoch99.pt` | `684,308.34` | `+76,955.09` |

完整指标已写入：

- `run/test_set2_november/metrics.md`
- `run/test_set2_november/metrics.csv`
- `run/test_set2_november/metrics.json`

test set2 结论和原 test split 不完全一致：原 test split 上完整三信号 `ScratchDispatchAwareRL` 最好；2023-11 历史窗口上 `ScratchPerceptionOnlyRL` 最好，`ScratchDispatchAwareNoShapingRL` 次之，`ScratchRL` 第三，完整三信号方法仍优于 Urgent 但不是第一。这说明 perception/shaping/cost 的最优组合对月份分布有敏感性，需要更多月份窗口验证后再判断跨时间泛化。

## 对上一轮方法的判断

上一轮保留在：

- `agent/reposition/rl_agent.py`
- `agent/reposition/dispatch_aware_rl_agent.py`
- `pretrain_reposition_policies.py`
- `agent/reposition/demand_gap_policy.py`
- `model/reposition/rl/`
- `model/reposition/dispatch_aware_rl/`

它们的实际训练路径是：

| 方法 | 初始权重来源 | 是否有 RL fine-tune | 当前保留模型 | 判断 |
|------|--------------|---------------------|--------------|------|
| `RL` | 需求缺口教师策略监督预训练 | 跑过短程 REINFORCE，但 epoch9 退化后删除 | `checkpoint_epoch0.pt` | 当前结果基本是教师策略蒸馏，不是从零 RL |
| `DispatchAwareRL` | 需求缺口教师策略监督预训练 | 跑过短程 REINFORCE + 感知奖励 fine-tune | `checkpoint_epoch9.pt` | 有 RL 调整，但建立在教师预训练上 |

因此，用户指出“只用了教师策略，实际上没有训练过 RL 吗”是成立的：旧 `RL` 尤其是这样；旧 `DispatchAwareRL` 有短程 RL fine-tune，但不是从随机初始化训练出的 RL。

旧方法已单独存档在 `archived_teacher_pretrain_rl_methods.md`，本轮不删除、不覆盖。

## 新增从零方法

当前从零训练 Scratch 系列方法：

| 方法 | 类 | 权重目录 | 是否使用教师标签 | 是否加载旧权重 |
|------|----|----------|------------------|----------------|
| `ScratchRL` | `ScratchRLRepositionAgent` | `model/reposition/scratch_rl/` | 否 | 否 |
| `ScratchDispatchAwareRL` | `ScratchDispatchAwareRLRepositionAgent` | `model/reposition/scratch_dispatch_aware_rl/` | 否 | 否 |
| `ScratchDispatchAwareNoCostRL` | `ScratchDispatchAwareNoCostRLRepositionAgent` | `model/reposition/scratch_dispatch_aware_no_cost_rl/` | 否 | 否 |
| `ScratchDispatchAwareNoShapingRL` | `ScratchDispatchAwareNoShapingRLRepositionAgent` | `model/reposition/scratch_dispatch_aware_no_shaping_rl/` | 否 | 否 |
| `ScratchCombinedRL` | `ScratchCombinedRLRepositionAgent` | `model/reposition/scratch_combined_rl/` | 否 | 否 |
| `ScratchCostOnlyRL` | `ScratchCostOnlyRLRepositionAgent` | `model/reposition/scratch_cost_only_rl/` | 否 | 否 |
| `ScratchPerceptionOnlyRL` | `ScratchPerceptionOnlyRLRepositionAgent` | `model/reposition/scratch_perception_only_rl/` | 否 | 否 |

入口已加入 `main_process.py`：

```powershell
python .\main_process.py --reposition_method ScratchRL
python .\main_process.py --reposition_method ScratchDispatchAwareRL
python .\main_process.py --reposition_method ScratchDispatchAwareNoCostRL
python .\main_process.py --reposition_method ScratchDispatchAwareNoShapingRL
python .\main_process.py --reposition_method ScratchCombinedRL
python .\main_process.py --reposition_method ScratchCostOnlyRL
python .\main_process.py --reposition_method ScratchPerceptionOnlyRL
```

训练脚本：

```powershell
.\run\ScratchRL\train.ps1
.\run\ScratchDispatchAwareRL\train.ps1
.\run\ScratchDispatchAwareNoCostRL\train.ps1
.\run\ScratchDispatchAwareNoShapingRL\train.ps1
.\run\ScratchCombinedRL\train.ps1
```

测试脚本：

```powershell
.\run\ScratchRL\test.ps1
.\run\ScratchDispatchAwareRL\test.ps1
.\run\ScratchDispatchAwareNoCostRL\test.ps1
.\run\ScratchDispatchAwareNoShapingRL\test.ps1
.\run\ScratchCombinedRL\test.ps1
```

## 状态特征

旧 RL 网络使用 8 维 action 特征。本轮 scratch 方法扩展为 12 维，并做显式归一化，避免网络直接面对尺度差异很大的原始数值。

每个候选厂站的原始 8 维特征来自：

```python
env.get_action_feature(sid, current_pid)
```

原始特征：

| 维度 | 名称 | 含义 |
|------|------|------|
| 0 | `dispatch_count` | 过去 60 分钟该厂站发车次数 |
| 1 | `truck_num` | 当前该厂站可用车辆数 |
| 2 | `distance` | 当前工地到该厂站距离 |
| 3 | `next_return` | 下一辆车预计返回该厂站的时间 |
| 4 | `returning_count` | 当前正在返回该厂站的车辆数 |
| 5 | `future_30` | 未来 30 分钟计划发车数 |
| 6 | `future_60` | 未来 60 分钟计划发车数 |
| 7 | `future_120` | 未来 120 分钟计划发车数 |

新增组合特征：

```text
future_need = future_30 + 0.5 * future_60 + 0.25 * future_120
supply = truck_num + returning_count
shortage = future_need - supply
shortage_ratio = shortage / (future_need + supply + 1.0)
```

最终 12 维输入：

| 维度 | 特征 |
|------|------|
| 0 | `dispatch_count / 10` |
| 1 | `truck_num / 20` |
| 2 | `distance / 30` |
| 3 | `next_return / 240` |
| 4 | `returning_count / 20` |
| 5 | `future_30 / 5` |
| 6 | `future_60 / 10` |
| 7 | `future_120 / 20` |
| 8 | `future_need / 10` |
| 9 | `supply / 20` |
| 10 | `shortage / 10` |
| 11 | `shortage_ratio` |

## 网络和算法

算法文件：

- `RLAlgo/scratch_policy_gradient.py`

网络：

```text
StationScoringNet(input_dim=12, hidden_dims=(128, 64), normalize=False)
```

输入输出：

```text
(batch, num_candidate_stations, 12) -> (batch, num_candidate_stations)
```

每个候选厂站输出一个分数，再用 softmax 得到动作概率。训练时采样，测试时贪心选择最大概率动作。

更新方式是 REINFORCE：

```text
G_t = r_t + gamma * G_{t+1}
loss = -log pi(a_t | s_t) * normalized(G_t)
```

稳定性处理：

- episode 内 return 标准化；
- entropy bonus，当前 `entropy_coef=0.02`；
- 梯度裁剪，`max_norm=5.0`；
- Adam 优化器。

关键点：这里没有 cross entropy 监督标签，也没有从 `model/reposition/rl/` 或 `model/reposition/dispatch_aware_rl/` 加载旧 checkpoint。

## 奖励设计

### ScratchRL

`ScratchRL` 使用两类奖励：

1. 成本增量奖励；
2. action-level shaping 奖励。

成本奖励：

```text
reward_cost = - delta_cost * (1 / 10000) / hours
```

其中成本包含：

- 重车去工地油耗；
- 空车返回油耗；
- 等待/空转油耗；
- 超时罚款；
- 连续浇筑中断罚款。

episode 结束时，厂站和工地加班补偿会均摊为负奖励。

当前 shaping：

```text
1.20 * low_stock_bonus
+ 4.00 * shortage_reduction
+ 0.45 * future_need
- 0.90 * truck_num
- 0.45 * returning_count
- 0.08 * distance
- 0.01 * next_return
- 0.03 * dispatch_count
```

这个 shaping 不来自教师动作标签，而是手工定义的环境奖励。它引导策略偏向车少、未来需求高、距离不过远的厂站。

### ScratchDispatchAwareRL

`ScratchDispatchAwareRL` 继承 `ScratchRL` 的成本和 shaping 奖励，并额外使用感知层奖励：

```text
perception_reward_scale = 1.0
```

感知层根据后续 dispatch 的 `intent_sid` 和 `actual_sid` 是否一致，把正负反馈写回此前的 reposition transition。

这仍然不是教师策略，因为它没有给定“应该选择哪个厂站”的标签；它只根据后续调度结果给策略梯度提供奖励。

### ScratchDispatchAwareRL 消融

为了确认 `ScratchDispatchAwareRL` 的效果到底来自哪些奖励信号，保留原 `ScratchRL` / `ScratchDispatchAwareRL` 方法和权重不动，新增两个独立消融方法：

| 消融方法 | 关闭项 | 保留项 | 权重目录 |
|----------|--------|--------|----------|
| `ScratchDispatchAwareNoCostRL` | 成本增量 reward、加班成本 reward | shaping + perception | `model/reposition/scratch_dispatch_aware_no_cost_rl/` |
| `ScratchDispatchAwareNoShapingRL` | action shaping reward | 成本增量 + 加班成本 + perception | `model/reposition/scratch_dispatch_aware_no_shaping_rl/` |
| `ScratchCostOnlyRL` | action shaping reward、perception reward | 成本增量 + 加班成本 | `model/reposition/scratch_cost_only_rl/` |
| `ScratchPerceptionOnlyRL` | 成本增量 reward、加班成本 reward、action shaping reward | perception | `model/reposition/scratch_perception_only_rl/` |

实现上使用三个 bool 开关：

```python
use_cost_reward = True
use_shaping_reward = True
use_perception_reward = True
```

默认 `ScratchRL` 为 `cost + shaping` 且 `use_perception_reward=False`；默认 `ScratchDispatchAwareRL` 为 `cost + shaping + perception`。各消融子类只改开关和权重目录。

训练脚本：

```powershell
.\run\ScratchDispatchAwareNoCostRL\train.ps1
.\run\ScratchDispatchAwareNoShapingRL\train.ps1
.\run\ScratchCostOnlyRL\train.ps1
.\run\ScratchPerceptionOnlyRL\train.ps1
```

测试脚本：

```powershell
.\run\ScratchDispatchAwareNoCostRL\test.ps1
.\run\ScratchDispatchAwareNoShapingRL\test.ps1
.\run\ScratchCostOnlyRL\test.ps1
.\run\ScratchPerceptionOnlyRL\test.ps1
```

训练日志确认开关生效：

| 方法 | 训练日志中的奖励分量 |
|------|----------------------|
| `ScratchDispatchAwareNoCostRL` | `cost_delta=0.0000`，`overtime=0.0000`；`shape` 和 `perception_raw` 非零 |
| `ScratchDispatchAwareNoShapingRL` | `shape=0.0000`；`cost_delta`、`overtime` 和 `perception_raw` 非零 |
| `ScratchCostOnlyRL` | `shape=0.0000`，`perception_raw=0.0000`；`cost_delta` 和 `overtime` 非零 |
| `ScratchPerceptionOnlyRL` | `base=0.0000`，`shape=0.0000`，`cost_delta=0.0000`，`overtime=0.0000`；`perception_raw` 非零 |

训练中验证结果：

| 方法 | validation 最优 epoch | validation 成本 RMB | 相对 validation Urgent |
|------|-----------------------|---------------------|------------------------|
| `ScratchDispatchAwareRL` | `99` | `1,314,243.65` | `-206,057.98` |
| `ScratchDispatchAwareNoCostRL` | `99` | `1,335,144.05` | `-185,157.58` |
| `ScratchDispatchAwareNoShapingRL` | `199` | `1,406,924.09` | `-113,377.54` |
| `ScratchCostOnlyRL` | `19` | `1,443,356.87` | `-76,944.76` |
| `ScratchPerceptionOnlyRL` | `19` | `1,355,021.16` | `-165,280.47` |

正式测试结果：

| 方法 | 默认加载 | test 成本 RMB | 相对 test Urgent | 相对 `ScratchDispatchAwareRL` |
|------|----------|---------------|------------------|-------------------------------|
| `ScratchDispatchAwareRL` | `checkpoint_epoch99.pt` | `916,819.31` | `-70,619.24` | baseline |
| `ScratchDispatchAwareNoCostRL` | `checkpoint_epoch99.pt` | `993,700.60` | `+6,262.05` | `+76,881.29` |
| `ScratchDispatchAwareNoShapingRL` | `checkpoint_epoch199.pt` | `989,989.44` | `+2,550.89` | `+73,170.13` |
| `ScratchCostOnlyRL` | `checkpoint_epoch19.pt` | `967,388.12` | `-20,050.43` | `+50,568.81` |
| `ScratchPerceptionOnlyRL` | `checkpoint_epoch19.pt` | `1,009,744.50` | `+22,305.95` | `+92,925.19` |

曲线图已生成：

| 方法 | cum reward 曲线 | validation cost 曲线 |
|------|-----------------|----------------------|
| `ScratchDispatchAwareNoCostRL` | `run/ScratchDispatchAwareNoCostRL/cum_reward_curve.png` | `run/ScratchDispatchAwareNoCostRL/validation_cost_curve.png` |
| `ScratchDispatchAwareNoShapingRL` | `run/ScratchDispatchAwareNoShapingRL/cum_reward_curve.png` | `run/ScratchDispatchAwareNoShapingRL/validation_cost_curve.png` |
| `ScratchCostOnlyRL` | `run/ScratchCostOnlyRL/cum_reward_curve.png` | `run/ScratchCostOnlyRL/validation_cost_curve.png` |
| `ScratchPerceptionOnlyRL` | `run/ScratchPerceptionOnlyRL/cum_reward_curve.png` | `run/ScratchPerceptionOnlyRL/validation_cost_curve.png` |

消融结论：

1. 只去掉成本 reward 后，validation 仍能低于 Urgent，但 test 成本变成 `993,700.60 RMB`，比 Urgent 高 `6,262.05 RMB`，泛化失败。
2. 只去掉 shaping 后，训练前期 validation 极差，直到 epoch199 才降到 `1,406,924.09 RMB`，最终 test 成本 `989,989.44 RMB`，仍比 Urgent 高 `2,550.89 RMB`。
3. 纯 cost 在原 test split 能超过 Urgent，但弱于 `ScratchRL`，说明 shaping 对 cost reward 的训练质量有帮助。
4. 纯 perception 在 validation 上能低于 Urgent，但原 test split 高于 Urgent；在 2023-11 test set2 上反而最低，说明单一 perception reward 的跨时间稳定性还需要更多月份验证。
5. 因此当前 `ScratchDispatchAwareRL` 超过 Urgent 的原 test split 表现不是纯 perception reward 造成的，而是 `cost + shaping + perception` 三者共同作用；其中 shaping 对训练早期稳定性尤其重要。

### ScratchCombinedRL

`ScratchCombinedRL` 仍然从随机初始化训练，权重单独保存在：

```text
model/reposition/scratch_combined_rl/
```

奖励结构是：

```text
combined_reward = ScratchRL成本/shaping奖励 + 1.1 * perception_raw_reward
```

系数 `1.1` 来自训练前短程校准，而不是手拍。校准命令使用临时目录、不开验证、只跑 20 episode，并把 `perception_reward_scale` 临时设为 `1.0`：

```powershell
python .\main_process.py --reposition_train_mode --reposition_method ScratchCombinedRL --reposition_episode_num 20 --save_model_frequency 999 --validation_frequency 0 --model_reposition_scratch_combined_save_dir model/reposition/scratch_combined_calibration_tmp --scratch_combined_perception_reward_scale 1.0
```

校准日志中 18 个有 reposition 的 episode 给出的量级是：

| 指标 | 数值 |
|------|------|
| `mean_abs_base` | `6,594.89` |
| `mean_abs_perception_raw` | `5,961.35` |
| `mean_abs_base / mean_abs_perception_raw` | `1.1063` |
| episode 比值中位数 | `1.5075` |

因此采用聚合均值比例附近的 `SCRATCH_COMBINED_PERCEPTION_REWARD_SCALE = 1.1`，使感知层奖励和成本/shaping 奖励在训练初期处于接近量级。训练日志中的 `Scratch reward stats` 会同时打印 `base`、`perception_raw` 和 `perception_scaled`，用于确认 `perception_scaled = 1.1 * perception_raw`。

正式训练使用和另外两个 Scratch 方法相同的周期验证与早停机制。当前 `best_checkpoint.json` 记录：

```json
{
  "best_epoch": 199,
  "validation_cost": 1239681.094836697,
  "reposition_method": "ScratchCombinedRL",
  "dispatch_method": "Fastest"
}
```

正式测试读取 `checkpoint_epoch199.pt`，测试成本为 `919,643.46 RMB`。这说明组合奖励版本超过 `Urgent` 和纯 `ScratchRL`，但没有超过当前最好的 `ScratchDispatchAwareRL`。因此当前排名仍应以 `ScratchDispatchAwareRL` 为最佳，`ScratchCombinedRL` 作为独立保留的组合奖励实验结果。

## 15 天历史实验训练命令

本轮训练前已清空 scratch 权重目录：

```powershell
Remove-Item .\model\reposition\scratch_rl\*.pt
Remove-Item .\model\reposition\scratch_dispatch_aware_rl\*.pt
```

15 天小范围实验的第二轮训练命令：

```powershell
python .\main_process.py --reposition_train_mode --reposition_method ScratchRL --reposition_episode_num 80 --save_model_frequency 10 --rl_lr 0.0005
python .\main_process.py --reposition_train_mode --reposition_method ScratchDispatchAwareRL --reposition_episode_num 80 --save_model_frequency 10 --rl_lr 0.0005
```

checkpoint 评估命令：

```powershell
python .\evaluate_scratch_rl_checkpoints.py
```

正式测试命令：

```powershell
python .\main_process.py --reposition_method Urgent
python .\main_process.py --reposition_method ScratchRL
python .\main_process.py --reposition_method ScratchDispatchAwareRL
```

## 15 天历史 checkpoint 评估结果

测试日期为 `2024-05-01` 到 `2024-05-15` 全量日期，dispatch 方法为 `Fastest`。

Urgent 基准：

| 方法 | 总成本 RMB |
|------|------------|
| `Urgent` | `1,405,603.55` |

ScratchRL：

| checkpoint | 总成本 RMB | 相对 Urgent |
|------------|------------|-------------|
| epoch 9 | `2,014,379.07` | `+608,775.52` |
| epoch 19 | `1,823,063.11` | `+417,459.56` |
| epoch 29 | `1,811,860.11` | `+406,256.56` |
| epoch 39 | `1,782,258.91` | `+376,655.36` |
| epoch 49 | `1,739,037.04` | `+333,433.49` |
| epoch 59 | `1,679,500.11` | `+273,896.57` |
| epoch 69 | `1,732,768.17` | `+327,164.62` |
| epoch 79 | `1,736,466.53` | `+330,862.98` |

ScratchRL 最优为 epoch 59，但仍差于 Urgent。

ScratchDispatchAwareRL：

| checkpoint | 总成本 RMB | 相对 Urgent |
|------------|------------|-------------|
| epoch 9 | `2,177,700.01` | `+772,096.47` |
| epoch 19 | `1,739,600.96` | `+333,997.41` |
| epoch 29 | `1,345,109.31` | `-60,494.24` |
| epoch 39 | `1,383,495.50` | `-22,108.04` |
| epoch 49 | `1,395,167.31` | `-10,436.24` |
| epoch 59 | `1,441,444.18` | `+35,840.64` |
| epoch 69 | `1,453,751.92` | `+48,148.37` |
| epoch 79 | `1,468,562.21` | `+62,958.66` |

ScratchDispatchAwareRL 最优为 epoch 29，比 Urgent 低 `60,494.24 RMB`。

## 15 天历史默认加载

15 天小范围实验中，因为最新 checkpoint 不一定最好，当时测试模式加载配置中的 best epoch：

```python
MODEL_REPOSITION_SCRATCH_RL_BEST_EPOCH = 59
MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_BEST_EPOCH = 29
```

训练模式仍然从最新 checkpoint 继续训练，避免影响后续实验。expanded split 的当前默认加载见本文开头“最新 expanded split 实验”。

正式测试确认：

| 方法 | 默认加载 | 总成本 RMB | 相对 Urgent |
|------|----------|------------|-------------|
| `Urgent` | 无 | `1,405,603.55` | baseline |
| `ScratchRL` | `checkpoint_epoch59.pt` | `1,679,500.11` | `+273,896.57` |
| `ScratchDispatchAwareRL` | `checkpoint_epoch29.pt` | `1,345,109.31` | `-60,494.24` |

## 15 天历史判断

1. 旧 `RL` / `DispatchAwareRL` 应归类为“教师策略预训练 + 少量 RL fine-tune”，不是严格意义上的从零 RL。
2. 在 15 天小范围全量口径下，`ScratchDispatchAwareRL checkpoint_epoch29.pt` 曾优于 Urgent。
3. 该 15 天结论已被 expanded split 替代；当前应以本文开头的 train/validation/test 结果为准。

后续如果继续提高 scratch 方法，优先方向应是降低延迟成本奖励方差，例如加入 value baseline / actor-critic，或把感知奖励改成更贴近最终成本的局部、可归因指标。
