两个Reposition方面的RL方法

# 基础RL方法

统计两次Reposition中的成本+惩罚，除以间隔时间，作为负的奖励项，以进行训练，训练采用REINFORCE方法

# Dispatch-Aware的Reposition RL方法

建立一个感知层，新增Dispatch Agent（目前是Fastest和Follow，对于任何dispatch的agent都一样）的输出，

- 原始意图Station：按照方法中理论上应该选择的Station
- 实际选择Station：可能和原始意图相符，也可能不符（由于Station没有更多车辆）

当Dispatch作出时，在感知层中记录

- 相符：原始和实际意图相等，作为正奖励
- 不符：原始和实际意图相悖，作为负奖励

以此指导Reposition RL方法的学习


# 感知层设计

建立一个感知层类并实例化，之后改造现有的Dispatch Agent方法（对以后任何的Dispatch Agent也是），均生成

- 原始意图Station：按照方法中理论上应该选择的Station，例如，Fastest应该选当前如果派车，最快到达工地的水泥厂站，但是也许该水泥厂站并没有足够的车辆。
- 实际选择Station：可能和原始意图相符，也可能不符（由于Station没有更多车辆）

实际最后反馈给主程序时，仍然返回实际选择Station，但是会在感知层中记录每一次的原始意图Station，Station当前的车辆数，其中多少是发出了又返回的车辆（需要你写一个逻辑往station里面记录一下返回的车辆有多少）实际选择Station，Station当前的车辆数，其中多少是发出了又返回的车辆（需要你写一个逻辑往station里面记录一下返回的车辆有多少），等等状态，这些状态会提供给Reposition Agent使用



此外，除了维护Dispatch的选择，感知层也要维护每次Reposition的选择，类似记录下来，因为要往回写入RL方法的奖励，当Dispatch作出选择的时候，应该会发生以下的事情：

- 原始和实际意图不符，这时候应该往之前的Reposition选择序列里面写入负的奖励，由于之前的Reposition可能有多次。设初始负奖励为-1，然后按照Reposition作出选择的时间，和当前Dispatch的时间的差值，使用因子进行缩放，当前代码使用的是分钟，先转换为小时，比如时间差值是，2h，那么，设定一个缩放因子  alpha，那么，分配给2h前的Reposition的负奖励就是 $(-1) \times \alpha^{2}$，alpha可以先设置为0.99，当然，这些参数都放到 @parameter里面，方便修改。
- 原始意图Station和实际意图相符，这时候应该判断发出的车是否来源于之前的Reposition，简单的方法是，判断Station当前车的数量（不-1）和Station记录的已返回的车辆数，如果后者大于等于，说明是来源于之前的Reposition，将正奖励 +1 写入到 之前的一个Reposition选择中（这个选择，首先，Reposition到了现在这个将要发出车的水泥Station，其次，是作出这些选择中最早的）。当一个Reposition选择被写入正奖励的时候，后续将不再能写入正奖励。
