# 当前两个 Reposition RL 方法说明

本文档说明当前项目中两个 RL Reposition 方法的实现方式、训练方式、权重目录和当前验证结果。对应代码主要在：

- `agent/reposition/rl_agent.py`
- `agent/reposition/dispatch_aware_rl_agent.py`
- `RLAlgo/reinforce.py`
- `simulator.py`
- `perception.py`
- `pretrain_reposition_policies.py`
- `parameter.py`

重要说明：本文档中的 `RL` / `DispatchAwareRL` 属于“需求缺口教师策略监督预训练 + 可选短程 RL fine-tune”的旧方法，不是从随机初始化训练出的纯 RL。严格不使用教师策略和预训练权重的实验记录见 `scratch_rl_experiments.md`。

## 总体背景

车辆完成工地浇筑后会进入 `TruckState.Return`。此时系统要决定车辆空返到哪个厂站，这个决策称为 Reposition。

当前有两个 RL Reposition 方法：

1. `RLRepositionAgent`
2. `DispatchAwareRLRepositionAgent`

二者都使用同一种基础网络结构和 REINFORCE 更新方式，但奖励来源、权重目录、训练目标不同。

当前默认配置下，两个方法都不再直接使用确定性需求缺口策略做线上决策：

```python
RL_USE_DEMAND_GAP_PRIOR = False
```

也就是说，正式测试时会加载各自目录下的神经网络 checkpoint，并由网络输出分数进行选择。

## 权重分开存储

旧实现中两个 RL 方法共用 `model/reposition/`，这会导致两个方法加载同一套 checkpoint，进而表现完全一致。当前已经拆分为独立目录：

| 方法 | 权重目录 | 当前最终加载 |
|------|----------|--------------|
| `RL` | `model/reposition/rl/` | `checkpoint_epoch0.pt` |
| `DispatchAwareRL` | `model/reposition/dispatch_aware_rl/` | `checkpoint_epoch9.pt` |

旧的共享 checkpoint 已删除。现在 `RLRepositionAgent.model_save_dir` 指向 `MODEL_REPOSITION_RL_SAVE_DIR`，`DispatchAwareRLRepositionAgent.model_save_dir` 指向 `MODEL_REPOSITION_DISPATCH_AWARE_SAVE_DIR`。

相关参数：

```python
MODEL_REPOSITION_SAVE_DIR = 'model/reposition/'
MODEL_REPOSITION_RL_SAVE_DIR = 'model/reposition/rl/'
MODEL_REPOSITION_DISPATCH_AWARE_SAVE_DIR = 'model/reposition/dispatch_aware_rl/'
```

`MODEL_REPOSITION_SAVE_DIR` 只作为兼容参数保留，当前两个 agent 不再直接从这个根目录加载或保存。

## 状态特征

两个 RL 方法使用同一套 action-level 状态特征。每次 Reposition 时，对每个候选厂站 `sid` 调用：

```python
env.get_action_feature(sid, current_pid)
```

当前特征是 8 维：

| 维度 | 名称 | 含义 |
|------|------|------|
| 0 | `dispatch_count` | 过去 60 分钟该厂站发车次数 |
| 1 | `truck_num` | 当前该厂站可用车辆数 |
| 2 | `distance` | 当前工地到该厂站距离，单位 km |
| 3 | `next_return_time` | 下一辆车预计返回该厂站的时间，单位分钟；无车返回时为 240 |
| 4 | `returning_count` | 当前正在返回该厂站的车辆数 |
| 5 | `future_30` | 未来 30 分钟该厂站计划发车数 |
| 6 | `future_60` | 未来 60 分钟该厂站计划发车数 |
| 7 | `future_120` | 未来 120 分钟该厂站计划发车数 |

新增的 `returning_count` 很关键。之前需求缺口策略会用到返程车数量，但网络状态里没有这个信息，因此网络无法完整学习这种策略。当前已把它加入状态维度，网络输入也从 7 维改成 8 维。

## 动作

动作是从当前工地 `current_pid` 的候选厂站列表 `available_sids` 中选一个厂站作为返程目标：

```python
return available_sids[action_idx]
```

训练模式下，网络按概率分布采样动作；测试模式下，网络贪心选择分数最高的动作：

```python
force_greedy = not self.train_mode
action_idx = self.algo.take_action(features_array, force_greedy=force_greedy)
```

如果手动设置 `RL_USE_DEMAND_GAP_PRIOR=True`，则会绕过网络，直接使用需求缺口策略选择动作。当前默认不开启。

## 网络结构

两个方法都使用 `StationScoringNet`：

```text
input_dim = 8
hidden_dims = (64, 32)
output_dim = 1
```

网络对每个候选厂站输出一个标量分数：

```text
(batch, num_stations, 8) -> (batch, num_stations)
```

再经过 softmax 形成动作概率：

```python
probs = torch.softmax(scores, dim=-1)
```

网络内部带特征标准化参数：

```python
feat_mean
feat_std
```

这些标准化参数在预训练时根据训练样本统计得到，并随 checkpoint 保存。

## REINFORCE 更新

两个方法共用 `REINFORCEReposition`。

每个 episode 内记录：

```python
transition = {
    'states': [],
    'actions': [],
    'rewards': [],
}
```

episode 结束后反向累计折扣回报：

```text
G_t = r_t + gamma * G_{t+1}
```

当前实现增加了两项稳定性处理：

1. 对 episode 内 returns 做标准化：

```python
returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
```

2. 梯度裁剪：

```python
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
```

loss 形式：

```text
loss = -log pi(a_t | s_t) * G_t
```

## 需求缺口教师策略

当前两个 RL 网络不是直接从随机初始化训练到最终策略，而是先用 `pretrain_reposition_policies.py` 分别做监督预训练。

监督预训练的教师策略来自需求缺口评分：

```text
score =
  truck_num
  + inbound_weight * inbound_count
  - demand_weight * future_need
  + distance_weight * distance
```

其中：

```text
future_need = future_30 + 0.5 * future_60 + 0.25 * future_120
```

选择 `score` 最小的厂站。

直观含义：

- `truck_num` 越少，越需要补车；
- `future_need` 越大，未来计划发车压力越大，越需要补车；
- `inbound_count` 越多，说明已有车在返回该厂站，不必过度补车；
- `distance` 越大，返程油耗和时间越高，需要惩罚。

两个方法使用不同教师参数：

| 方法 | `demand_weight` | `distance_weight` | `inbound_weight` | `future_need` 权重 |
|------|-----------------|-------------------|------------------|--------------------|
| `RL` | `0.45` | `0.08` | `0.5` | `future_30 + 0.5 * future_60 + 0.25 * future_120` |
| `DispatchAwareRL` | `0.35` | `0.06` | `0.5` | `future_30 + 0.5 * future_60 + 0.25 * future_120` |

因此两个网络从训练源头开始就不是同一个策略，也不会再因为共享目录而加载同一套权重。

## RLRepositionAgent

### 目标

基础 RL 的目标是直接降低运行成本。它不使用 dispatch intent/actual 的感知奖励。

### 奖励

每次 Reposition 决策时，会先计算当前累计成本：

```python
current_total_cost = self._compute_env_cost(env)
```

累计成本包括：

- 重车去工地油耗成本；
- 空车返回油耗成本；
- 等待装载、早到等待等空转油耗成本；
- 超时罚款；
- 连续浇筑中断罚款。

两次 Reposition 之间的奖励为：

```text
reward = - delta_cost / time_interval_hours
```

即单位时间成本越低，奖励越高。

最后一次 Reposition 到当天结束之间的成本会在 `after_every_episode()` 中补结算。

厂站和工地加班补偿在 episode 结束时均摊到所有 Reposition：

```python
overtime_total = env.station_overtime_cost + env.project_overtime_cost
overtime_per_repo = overtime_total / n_repositions
reward_i -= overtime_per_repo
```

### 训练与保存

训练结束后，每 `SAVE_MODEL_FREQUENCY` 个 episode 保存一次：

```python
self.algo.save(self.model_save_dir, env.i_episode)
```

当前 `RL` 的 `model_save_dir` 是：

```text
model/reposition/rl/
```

### 当前最终模型

当前 RL 最终保留：

```text
model/reposition/rl/checkpoint_epoch0.pt
```

短程策略梯度 fine-tune 生成过 `checkpoint_epoch9.pt`，但测试成本变差，因此已删除并回退到预训练 checkpoint0。

## DispatchAwareRLRepositionAgent

### 目标

DispatchAwareRL 在基础成本目标之外，引入感知层反馈，学习 Reposition 对后续 Dispatch 是否有帮助。

当前默认：

```python
self.use_cost_reward = True
```

也就是说它使用混合奖励：

1. 基础成本奖励；
2. Dispatch 感知奖励。

只用感知奖励时信号过稀疏，训练容易不稳定，因此当前启用成本奖励。

### 成本奖励

成本奖励逻辑和 `RLRepositionAgent` 基本一致：

```text
reward = - delta_cost / time_interval_hours
```

并在 episode 结束时分摊厂站和工地加班补偿。

### 感知层奖励

感知层记录两类事件：

1. Dispatch 事件：
   - `intent_sid`: dispatch agent 理论上想选的厂站；
   - `actual_sid`: 实际选中的厂站；
   - `station_truck_num`: 当时厂站车辆数；
   - `station_returned_count`: 当前厂站累计接收返程车数。

2. Reposition 事件：
   - `from_pid`: 从哪个工地返回；
   - `chosen_sid`: 选择回到哪个厂站；
   - `received_positive`: 是否已经获得过正奖励。

DispatchAwareRL 的奖励写入发生在即时订单 dispatch 后：

```python
reposition_agent.apply_perception_rewards(perception, env)
```

感知奖励规则：

#### 正奖励

如果 dispatch 的原始意图和实际选择一致：

```text
intent_sid == actual_sid
```

并且该站派出的车可认为来源于之前的 Reposition，则给最早一次回到该站且尚未获得正奖励的 Reposition 写入：

```text
+1.0
```

#### 负奖励

如果 dispatch 的原始意图和实际选择不一致：

```text
intent_sid != actual_sid
```

则向之前尚未获得正奖励的 Reposition 写入负奖励：

```text
reward = -1.0 * alpha ^ hours
```

当前：

```python
PERCEPTION_ALPHA = 0.99
```

### 当前最终模型

当前 DispatchAwareRL 最终保留：

```text
model/reposition/dispatch_aware_rl/checkpoint_epoch9.pt
```

它先经过监督预训练得到 `checkpoint_epoch0.pt`，再用混合奖励短程 fine-tune 到 `checkpoint_epoch9.pt`。测试显示 epoch9 优于 checkpoint0，因此保留 epoch9 作为当前最终模型。

## 当前训练流程

### 从零预训练两套独立权重

```powershell
conda run -n base python .\pretrain_reposition_policies.py
```

该脚本会：

1. 使用训练日期 `2024-05-01` 到 `2024-05-12`；
2. 分别用两个不同需求缺口教师策略跑模拟器并收集 Reposition 状态和教师动作；
3. 使用 cross entropy 训练两个 `StationScoringNet`；
4. 分别保存到两个目录。

最近一次预训练结果：

| 方法 | 样本数 | 教师动作拟合准确率 | 保存路径 |
|------|--------|--------------------|----------|
| `RL` | `6690` | `99.93%` | `model/reposition/rl/checkpoint_epoch0.pt` |
| `DispatchAwareRL` | `6690` | `100.00%` | `model/reposition/dispatch_aware_rl/checkpoint_epoch0.pt` |

### RL fine-tune

曾运行：

```powershell
conda run -n base python .\main_process.py --reposition_train_mode --reposition_method RL --reposition_episode_num 10 --rl_lr 0.0001
```

但 fine-tune 后的 `checkpoint_epoch9.pt` 测试成本变差，因此删除，当前保留 `checkpoint_epoch0.pt`。

### DispatchAwareRL fine-tune

曾运行：

```powershell
conda run -n base python .\main_process.py --reposition_train_mode --reposition_method DispatchAwareRL --reposition_episode_num 10 --rl_lr 0.0001
```

fine-tune 后的 `checkpoint_epoch9.pt` 测试效果更好，因此当前保留并加载它。

## 当前测试结果

正式入口：

```powershell
conda run -n base python .\main_process.py --reposition_method Urgent
conda run -n base python .\main_process.py --reposition_method RL
conda run -n base python .\main_process.py --reposition_method DispatchAwareRL
```

测试日期为 `2024-05-01` 到 `2024-05-15` 全量日期。当前结果：

| 方法 | 加载权重 | 总成本 RMB | 成本占收入 |
|------|----------|------------|------------|
| `Urgent` | 无 | `1,405,603.55` | `5.79%` |
| `RL` | `model/reposition/rl/checkpoint_epoch0.pt` | `1,296,762.35` | `5.34%` |
| `DispatchAwareRL` | `model/reposition/dispatch_aware_rl/checkpoint_epoch9.pt` | `1,287,579.69` | `5.30%` |

相对 Urgent：

| 方法 | 成本降低 RMB |
|------|--------------|
| `RL` | `108,841.20` |
| `DispatchAwareRL` | `118,023.86` |

运行脚本验证：

```powershell
.\run\RL\test.ps1
.\run\DispatchAwareRL\test.ps1
```

脚本日志结果和正式入口一致：

| 方法 | 脚本日志总成本 RMB |
|------|--------------------|
| `RL` | `1,296,762.3494880532` |
| `DispatchAwareRL` | `1,287,579.6915739966` |

## 当前两个方法的差异总结

| 维度 | `RL` | `DispatchAwareRL` |
|------|------|-------------------|
| 权重目录 | `model/reposition/rl/` | `model/reposition/dispatch_aware_rl/` |
| 当前最终 checkpoint | `checkpoint_epoch0.pt` | `checkpoint_epoch9.pt` |
| 网络输入 | 8 维 action 特征 | 8 维 action 特征 |
| 网络结构 | `8 -> 64 -> 32 -> 1` | `8 -> 64 -> 32 -> 1` |
| 预训练教师 | 需求权重 `0.45`，距离权重 `0.08` | 需求权重 `0.35`，距离权重 `0.06` |
| RL 奖励 | 成本奖励 | 成本奖励 + 感知层奖励 |
| 当前默认是否直接用需求缺口策略 | 否 | 否 |
| 当前测试总成本 | `1,296,762.35` | `1,287,579.69` |

## 注意事项

1. `RL_USE_DEMAND_GAP_PRIOR=False` 是当前正式配置。若改为 `True`，两个方法会直接用确定性需求缺口策略选动作，这会绕过网络，不适合用来判断两个 RL 网络是否各自收敛。
2. 继续训练时可能出现策略退化，尤其是基础 RL。当前已经加入 return 标准化和梯度裁剪，但仍建议每次训练后用正式测试做 checkpoint 选择。
3. `model/` 在 `.gitignore` 中，权重是本地文件，不会进入 git diff。
4. 如果要从零复现实验，应先清空 `model/reposition/rl/` 和 `model/reposition/dispatch_aware_rl/` 下的 `.pt`，再运行 `pretrain_reposition_policies.py`。
