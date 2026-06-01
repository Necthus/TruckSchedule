两个Reposition方面的RL方法

# 基础RL方法

统计两次Reposition中的成本+惩罚，除以间隔时间，作为负的奖励项，以进行训练

# Dispatch-Aware的Reposition RL方法

建立一个感知层，新增Dispatch Agent（目前是Fastest和Follow，对于任何dispatch的agent都一样）的输出，

- 原始意图Station：按照方法中理论上应该选择的Station
- 实际选择Station：可能和原始意图相符，也可能不符（由于Station没有更多车辆）

当Dispatch作出时，在感知层中记录

- 相符：原始和实际意图相等，作为正奖励
- 不符：原始和实际意图相悖，作为负奖励


以此指导Reposition RL方法的学习
