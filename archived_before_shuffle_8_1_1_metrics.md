# 周期验证顺序训练结果归档

本文档归档切换到“默认打乱训练 + 更长训练集”之前的 scratch 权重和指标。

## 归档位置

权重和训练日志已复制到：

```text
model/reposition/archive_before_shuffle_8_1_1/
```

包含：

- `scratch_rl/`
- `scratch_dispatch_aware_rl/`
- `scratch_rl_train.log`
- `scratch_dispatch_aware_rl_train.log`

## 旧配置

```text
EXPERIMENT_START_DATE = 2024-03-01
EXPERIMENT_END_DATE = 2024-06-04
TRAIN_TEST_SPLIT_RATIO = 0.70
VALIDATION_SPLIT_RATIO = 0.15
VALIDATION_FREQUENCY = 20
EARLY_STOP_PATIENCE = 5
```

日期切分：

| split | 日期数 | 日期范围 |
|-------|--------|----------|
| train | 67 | `2024-03-01 ~ 2024-05-06` |
| validation | 14 | `2024-05-07 ~ 2024-05-20` |
| test | 15 | `2024-05-21 ~ 2024-06-04` |

训练日期按时间顺序循环，未打乱。

## 旧 best checkpoint

| 方法 | best epoch | validation cost RMB |
|------|------------|---------------------|
| `ScratchRL` | `139` | `1,439,358.62` |
| `ScratchDispatchAwareRL` | `99` | `1,370,207.42` |

## 旧 test split 指标

测试集为 `2024-05-21 ~ 2024-06-04`，dispatch 方法为 `Fastest`。

| 方法 | 加载权重 | test cost RMB | 相对 Urgent |
|------|----------|---------------|-------------|
| `Urgent` | 无 | `987,438.55` | baseline |
| `ScratchRL` | `checkpoint_epoch139.pt` | `978,449.69` | `-8,988.87` |
| `ScratchDispatchAwareRL` | `checkpoint_epoch99.pt` | `1,082,014.43` | `+94,575.87` |

## 说明

此归档只作为对照。后续实验会保持 validation/test 日期不变，将训练集向前扩展并默认打乱训练日期。
