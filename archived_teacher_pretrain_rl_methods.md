# 教师策略预训练 RL 方法存档

本文档存档上一轮基于需求缺口教师策略和监督预训练的两个 Reposition RL 方法。该方法先保留，不作为本轮“从零训练 RL”的目标实现。

## 方法状态

上一轮方法包含：

- `RLRepositionAgent`
- `DispatchAwareRLRepositionAgent`
- `pretrain_reposition_policies.py`
- `agent/reposition/demand_gap_policy.py`

它们的核心特征是：

1. 使用 8 维 action-level 特征；
2. 用需求缺口教师策略生成动作标签；
3. 用 cross entropy 对 `StationScoringNet` 做监督预训练；
4. `DispatchAwareRL` 在预训练后额外做了短程 REINFORCE fine-tune；
5. 两个方法的权重已分开存储。

## 权重目录

| 方法 | 权重目录 | 上一轮最终 checkpoint |
|------|----------|-----------------------|
| `RL` | `model/reposition/rl/` | `checkpoint_epoch0.pt` |
| `DispatchAwareRL` | `model/reposition/dispatch_aware_rl/` | `checkpoint_epoch9.pt` |

## 上一轮正式测试结果

测试日期为 `2024-05-01` 到 `2024-05-15` 全量日期。

| 方法 | 总成本 RMB | 成本占收入 |
|------|------------|------------|
| `Urgent` | `1,405,603.55` | `5.79%` |
| `RL` | `1,296,762.35` | `5.34%` |
| `DispatchAwareRL` | `1,287,579.69` | `5.30%` |

## 本轮不再沿用的部分

本轮从零训练 RL 时不使用：

- 需求缺口教师动作标签；
- 监督预训练权重；
- `model/reposition/rl/` 和 `model/reposition/dispatch_aware_rl/` 中的已有 checkpoint；
- `RL_USE_DEMAND_GAP_PRIOR=True` 这种绕过网络的确定性选择。

本轮新增方法会使用新的独立目录，避免覆盖或污染上一轮权重。
