# 项目改造调整文档

## 改造总览

根据 `adjust.md` 和 `method_design.md` 的要求，对混凝土调度项目进行了三项主要改造：**训练/测试模式**、**感知层**、**RL Reposition Agent**。

---

## 一、训练/测试模式

### 新增参数（parameter.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `REPOSITION_TRAIN_MODE` | `False` | 是否训练Reposition Agent |
| `DISPATCH_TRAIN_MODE` | `False` | 是否训练Dispatch Agent（预留） |
| `TRAIN_TEST_SPLIT_RATIO` | `0.8` | 训练集日期比例，剩余为测试集 |
| `TEST_REPEAT_NUM` | `1` | 测试模式下的重复轮数 |
| `REPOSITION_EPISODE_NUM` | `2000` | 训练episode总数 |
| `SAVE_MODEL_FREQUENCY` | `10` | 每隔多少episode保存一次模型 |
| `RL_LR` | `0.001` | REINFORCE学习率 |
| `RL_GAMMA` | `0.99` | 折扣因子 |
| `MODEL_REPOSITION_SAVE_DIR` | `model/reposition/` | Reposition模型保存目录 |

### 日期分割逻辑（main_process.py）

- **分割方式：** 将实验日期段（`EXPERIMENT_START_DATE` ~ `EXPERIMENT_END_DATE`）按 `TRAIN_TEST_SPLIT_RATIO` 切分，前段用于训练，后段用于测试
- **训练循环：** 参照 MobRef 设计，训练模式下循环 train_dates 日期池，每个 episode 取一天。日期池循环复用直至达到 `REPOSITION_EPISODE_NUM`
- **测试循环：** 训练结束后自动切换 test_dates 进行测试评估；纯测试模式则遍历全部日期
- **代码结构：** 提取 `run_one_day()` 函数封装单日模拟逻辑，训练/测试共享同一套流程，仅数据来源不同

### Agent生命周期（agent/reposition/base_agent.py）

参照 MobRef 的 BaseAgent 模式，为 BaseRepositionAgent 新增统一接口：

```
model_initialization()   → 模型加载/初始化，返回起始episode编号
before_every_episode()   → 每回合开始前准备
after_every_episode()    → 每回合结束后收尾（含RL更新、模型保存）
after_every_step()       → 每步结束后收尾
```

---

## 二、感知层

### 新增文件：perception.py

感知层 `PerceptionLayer` 类维护三类记录，为Dispatch-Aware RL提供奖励信号。

#### DispatchPerceptionRecord（调度记录）

每次Dispatch发生时记录以下信息：

| 字段 | 说明 |
|------|------|
| `intent_sid` | Dispatch Agent的原始意图厂站（理论上应选的站） |
| `actual_sid` | 实际选择的厂站（可能因车辆不足而不同） |
| `station_truck_num` | 实际选择厂站当时的车辆数 |
| `station_returned_count` | 实际选择厂站累计返回车辆数 |

#### RepositionPerceptionRecord（重定位记录）

每次Reposition发生时记录：

| 字段 | 说明 |
|------|------|
| `from_pid` | 从哪个工地返回 |
| `chosen_sid` | 选择回到哪个厂站 |
| `available_sids` | 当时可选的厂站列表 |
| `received_positive` | 是否已获得正奖励（标记位，防重复） |

#### 奖励分配逻辑

**负奖励（意图不符时）：**

当Dispatch Agent的原始意图与实际选择不一致时，向**该时刻之前所有**的Reposition记录写入负奖励：

```
单个负奖励 = (-1) × α^(hours)
α = PERCEPTION_ALPHA（默认0.99）
hours = 当前Dispatch时间与Reposition时间之差（小时）
```

时间越久远的Reposition，惩罚被指数衰减稀释。

**正奖励（意图相符时）：**

当Dispatch的原始意图与实际选择一致时，检查发出车辆的厂站是否有足够多的返回车辆（`returned_truck_count >= truck_num + 1`）。如果该车来源于之前的Reposition，则找到**最早**一个Reposition到该厂站的记录，写入 `+1` 奖励。每条Reposition记录最多获得一次正奖励。

### 配套修改：component.py — Station类

新增字段/方法：

| 新增项 | 说明 |
|--------|------|
| `returned_truck_count` | 每天重置的累计返回车辆数，`receive_truck()` 时+1 |
| `dispatched_from_returned` | 从返回车辆中派出的次数 |
| `dispatch_came_from_reposition()` | 判断当前要派出的车是否来源于之前的reposition |

### 配套修改：Dispatch Agent（fastest_agent.py / follow_agent.py）

每个Dispatch Agent拆分为两个方法：

- **`get_intent_station()`** — 返回原始意图厂站：
  - Fastest：理论最快到达的站（不考虑该站是否有车）
  - Follow：跟随上次服务的站
- **`select_a_station()`** — 返回实际可选厂站：
  - Fastest：在**有车**的站中选最快的
  - Follow：跟随上次服务的站（与意图一致）

调用方（main_process.py）分别调用两个方法，将结果记录到感知层。

---

## 三、RL Reposition Agent

### 新增文件：RLAlgo/reinforce.py

`REINFORCEReposition` 类 — 标准的REINFORCE策略梯度算法。

**网络结构：** `StationScoringNet`
- 输入：每个可用厂站的7维特征向量
- 输出：每个厂站的标量分数 → Softmax → 动作概率分布
- 训练时按概率采样，测试时贪心选择

**状态特征（7维）：**

| 维度 | 含义 |
|------|------|
| 0 | 过去1小时发车数量 |
| 1 | 当前可用车数 |
| 2 | 距离工地的距离 |
| 3 | 下一辆车返回时间 |
| 4 | 未来30分钟计划发车数 |
| 5 | 未来60分钟计划发车数 |
| 6 | 未来120分钟计划发车数 |

**更新逻辑：** 从后向前累积折扣回报 G，每一步计算 `loss = -log_prob(action) × G`，累积梯度后统一更新。

**Checkpoint：** 保存/加载 `model_state_dict` + `optimizer_state_dict`，支持断点续训。

### 新增文件：agent/reposition/rl_agent.py

`RLRepositionAgent` — 基础RL Reposition方法。

- **奖励设计：** 统计两次Reposition之间的总成本（油耗+惩罚），除以间隔时间（小时），取负作为奖励项。即鼓励低成本高效率的Reposition选择。
- **训练流程：**
  1. 每次Reposition时，先结算上一段的负奖励写入transition
  2. 构建厂站特征 → 调用RL算法选择动作
  3. 记录(state, action)到transition
  4. Episode结束后调用 `algo.update()` 执行REINFORCE更新
  5. 按频率保存模型

### 新增文件：agent/reposition/dispatch_aware_rl_agent.py

`DispatchAwareRLRepositionAgent` — Dispatch感知的RL方法。

- **继承基础RL**的全部逻辑（成本奖励）
- **额外奖励来源：** 通过 `apply_perception_rewards()` 方法在每次Dispatch发生后调用，从感知层读取正负奖励信号，追加到对应Reposition的transition中
- **实现机制：**
  1. Dispatch发生后，调用 `perception.process_dispatch_result()`
  2. 获得正奖励列表 `[(repo_idx, +1), ...]` 和负奖励列表 `[(repo_idx, -α^hours), ...]`
  3. 按索引追加到 `algo.transition['rewards'][repo_idx]`
  4. 这使得Reposition的选择会同时考虑成本和Dispatch反馈

---

## 四、文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `parameter.py` | 新增训练/测试模式、RL、感知层参数 |
| 修改 | `component.py` | Station新增返回车辆计数及判断方法 |
| 修改 | `simulator.py` | 新增 `i_episode` 属性 |
| 修改 | `agent/dispatch/base_agent.py` | 新增 `get_intent_station()` 抽象接口 |
| 修改 | `agent/dispatch/fastest_agent.py` | 分离 `get_intent_station()` 和 `select_a_station()` |
| 修改 | `agent/dispatch/follow_agent.py` | 分离 `get_intent_station()` 和 `select_a_station()` |
| 修改 | `agent/reposition/base_agent.py` | 新增 `train_mode` 参数和RL生命周期钩子 |
| 修改 | `agent/reposition/urgent_agent.py` | 适配新基类签名 |
| 修改 | `agent/reposition/retrace_agent.py` | 适配新基类签名 |
| 修改 | `main_process.py` | 重构为训练/测试双模式，集成感知层 |
| 新增 | `perception.py` | 感知层实现 |
| 新增 | `RLAlgo/__init__.py` | 包初始化 |
| 新增 | `RLAlgo/reinforce.py` | REINFORCE算法实现 |
| 新增 | `network/__init__.py` | 包初始化 |
| 新增 | `network/base_net.py` | 策略网络（MLP + StationScoringNet） |
| 新增 | `agent/reposition/rl_agent.py` | 基础RL Reposition Agent |
| 新增 | `agent/reposition/dispatch_aware_rl_agent.py` | Dispatch感知RL Agent |

---

## 五、使用方式

```powershell
# 传统方法测试
python main_process.py --reposition_method Retrace

# 传统方法对比
python main_process.py --reposition_method Urgent

# RL训练模式
python main_process.py --reposition_train_mode True --reposition_method RL

# Dispatch感知RL训练
python main_process.py --reposition_train_mode True --reposition_method DispatchAwareRL

# 调整训练集比例
python main_process.py --train_test_split_ratio 0.7

# 修改训练episode数
python main_process.py --reposition_train_mode True --reposition_method RL --reposition_episode_num 500

# 查看所有参数
python main_process.py --help
```
