import os
import math
# 随机性参数
SEED = 2333
MAGIC_MAX_NUM  = 233333

# 运行设备参数
GPU_ID = 0

# 方法参数

DISPATCH_METHOD = 'Fastest' # 派单方法 可选项：'Fastest'（优先派送最快的车） 'Follow'（跟随上次选择的厂站）
REPOSITION_METHOD = 'Retrace' # 车辆调度方法 可选项：'Urgent' 'Retrace' 'RL' 'DispatchAwareRL' 'ScratchRL' 'ScratchDispatchAwareRL' 'ScratchDispatchAwareNoCostRL' 'ScratchDispatchAwareNoShapingRL' 'ScratchCombinedRL' 'ScratchCostOnlyRL' 'ScratchPerceptionOnlyRL'

# 训练/测试模式参数
REPOSITION_TRAIN_MODE = False  # 是否训练Reposition Agent
DISPATCH_TRAIN_MODE = False    # 是否训练Dispatch Agent（预留）
TRAIN_TEST_SPLIT_RATIO = 0.8   # 训练集比例
VALIDATION_SPLIT_RATIO = 0.1   # 验证集比例，剩余日期作为测试集
SHUFFLE_TRAIN_DATES = True     # 训练时是否按cycle打乱训练日期顺序

# 测试重复次数
TEST_REPEAT_NUM = 1

# RL算法参数
RL_LR = 0.001           # 学习率
RL_GAMMA = 0.99         # 折扣因子
RL_EPSILON = 0.1        # epsilon-greedy探索率
RL_EPSILON_MIN = 0.01   # 最小探索率
RL_EPSILON_DECAY = 0.999  # epsilon衰减率

# RL Reposition需求缺口先验参数。
# 默认不直接接管线上决策；可在实验/引导训练时通过 --rl_use_demand_gap_prior 开启。
RL_USE_DEMAND_GAP_PRIOR = False
RL_DEMAND_GAP_DEMAND_WEIGHT = 0.45
RL_DEMAND_GAP_DISTANCE_WEIGHT = 0.08
RL_DEMAND_GAP_INBOUND_WEIGHT = 0.5
RL_DEMAND_GAP_INBOUND_HORIZON = 999
RL_DEMAND_GAP_FUTURE_30_WEIGHT = 1.0
RL_DEMAND_GAP_FUTURE_60_WEIGHT = 0.5
RL_DEMAND_GAP_FUTURE_120_WEIGHT = 0.25

# RL训练参数
REPOSITION_EPISODE_NUM = 2000   # 训练episode数（仅训练模式）
SAVE_MODEL_FREQUENCY = 10       # 每隔多少episode保存一次模型
VALIDATION_FREQUENCY = 20       # 训练时每隔多少episode在验证集评估一次
EARLY_STOP_PATIENCE = 5         # 验证集连续多少次无提升后早停；<=0表示不早停
EARLY_STOP_MIN_DELTA = 0.0      # 验证集成本至少降低多少才算提升
MODEL_SAVE_DIR = 'model'
MODEL_REPOSITION_SAVE_DIR = 'model/reposition/'
MODEL_REPOSITION_RL_SAVE_DIR = 'model/reposition/rl/'
MODEL_REPOSITION_DISPATCH_AWARE_SAVE_DIR = 'model/reposition/dispatch_aware_rl/'
MODEL_REPOSITION_SCRATCH_RL_SAVE_DIR = 'model/reposition/scratch_rl/'
MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_SAVE_DIR = 'model/reposition/scratch_dispatch_aware_rl/'
MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_COST_SAVE_DIR = 'model/reposition/scratch_dispatch_aware_no_cost_rl/'
MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_SHAPING_SAVE_DIR = 'model/reposition/scratch_dispatch_aware_no_shaping_rl/'
MODEL_REPOSITION_SCRATCH_COMBINED_SAVE_DIR = 'model/reposition/scratch_combined_rl/'
MODEL_REPOSITION_SCRATCH_COST_ONLY_SAVE_DIR = 'model/reposition/scratch_cost_only_rl/'
MODEL_REPOSITION_SCRATCH_PERCEPTION_ONLY_SAVE_DIR = 'model/reposition/scratch_perception_only_rl/'
MODEL_REPOSITION_SCRATCH_RL_BEST_EPOCH = 99
MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_BEST_EPOCH = 99
MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_COST_BEST_EPOCH = 99
MODEL_REPOSITION_SCRATCH_DISPATCH_AWARE_NO_SHAPING_BEST_EPOCH = 199
MODEL_REPOSITION_SCRATCH_COMBINED_BEST_EPOCH = 199
MODEL_REPOSITION_SCRATCH_COST_ONLY_BEST_EPOCH = 99
MODEL_REPOSITION_SCRATCH_PERCEPTION_ONLY_BEST_EPOCH = 99
MODEL_DISPATCH_SAVE_DIR = 'model/dispatch/'

# Scratch combined reward参数。
# ScratchCombinedRL使用ScratchRL的成本/shape奖励，并按该系数叠加感知层原始奖励。
SCRATCH_COMBINED_PERCEPTION_REWARD_SCALE = 1.1

# 感知层参数
PERCEPTION_ALPHA = 0.99  # 时间衰减因子，用于分配负奖励

# 地球参数

ENV_MIN_LAT = 22.4
ENV_MAX_LAT = 23.0

ENV_MIN_LON = 113.7
ENV_MAX_LON = 114.4

GRID_SIZE_KM = 1.0  # 网格大小，单位：公里


KM_PER_DEGREE_LAT = 111.0  # 每度纬度约等于111公里，地球是椭球体，实际从赤道的110.567公里（68.703英里）到极地的111.699公里（69.41英里）不等
KM_PER_DEGREE_LON = 102.40172792964334
# 实际采用下式进行计算
# KM_PER_DEGREE_LON = KM_PER_DEGREE_LAT * math.cos(math.radians((ENV_MIN_LAT + ENV_MAX_LAT) / 2))  # 经度每度公里数随纬度变化而变化


# 实验验证参数

IGNORE_INSTANT_ORDERS = False  # 是否忽略实时订单（只验证计划订单的调度效果）
ONLY_KEEP_ACTIVE_STATION = True

# 物理环境现实参数

TRUCK_SPEED = 0.5 # 车辆速度 km/minute 即30km/h

LOAD_CONCRETE_TIME = 10 # 分钟
UNLOAD_CONCRETE_TIME = 5 # 浇筑混凝土时间 分钟


# 耗油

LOAD_GO_FUEL_CONSUMPTION_PER_KM = 0.5 # 车辆空载行驶每公里的油耗（单位：升/公里）
EMPTY_RETURN_FUEL_CONSUMPTION_PER_KM = 0.3 # 车辆空载返回每公里的油耗（单位：升/公里）

LOAD_WAITING_FUEL_CONSUMPTION_PER_MINUTE = 0.1 # 车辆等待装载每分钟的油耗（单位：升/分钟）

OIL_PRICE = 8 # 每升油的价格（单位：元/升）

# 和钱有关的参数

CONCRETE_REVENUE_PER_FANG = 350 # 每方混凝土的收入

OVERTIME_PAY_FOR_DRIVER_PER_MINUTE = 1 # 司机加班每分钟的工资（单位：元/分钟）







OVERTIME_PAY_FOR_STATION_PER_MINUTE = 20*OVERTIME_PAY_FOR_DRIVER_PER_MINUTE # 厂站加班每分钟的补偿（单位：元/分钟）
OVERTIME_PAY_FOR_PROJECT_PER_MINUTE = 20*OVERTIME_PAY_FOR_DRIVER_PER_MINUTE # 工地加班每分钟的补偿（单位：元/分钟）


# 初始化环境参数

STATION_INIT_TRUCK_NUM = 15  # 每个站点初始车辆数
STATION_PRODUCTION_LINE_SIZE = 4  # 每个站点生产线数量


# 订单时间限制


ORDER_START_HOUR = 7  # 订单允许的最早开始时间（小时）
ORDER_END_HOUR = 20  # 订单允许的最晚结束时间（小时）

EXPERIMENT_START_DATE = '2024-01-12'  # 实验开始日期
EXPERIMENT_END_DATE = '2024-06-04'    # 实验结束日期



# 模拟器参数

WITH_ORIGIN_INTERACTION = True  # 是


import argparse
import inspect

def auto_generate_args():
    """自动扫描当前模块中的大写变量，生成对应的argparse参数"""
    parser = argparse.ArgumentParser()
    
    # 获取当前模块的所有变量
    current_module = inspect.currentframe().f_back.f_globals
    
    # 筛选出大写变量（通常宏变量用全大写命名）
    uppercase_vars = {
        name: value for name, value in current_module.items()
        if name.isupper() and not name.startswith('_')  # 排除内置变量
    }
    
    # 为每个大写变量自动生成add_argument
    for var_name, default_value in uppercase_vars.items():
        # 转换变量名为命令行参数名（全大写转小写，如LR→--lr）
        arg_name = f"--{var_name.lower()}"
        
        # 自动推断参数类型（根据默认值）
        arg_type = type(default_value)
        
        # 生成帮助信息（可自定义格式）
        help_msg = f"{var_name}的默认值: {default_value}"
        
        # 特殊处理布尔类型（避免命令行传入值的问题）
        if arg_type is bool:
            # 布尔参数用action='store_true'/'store_false'
            if default_value is False:
                parser.add_argument(arg_name, action='store_true', help=help_msg)
            else:
                parser.add_argument(arg_name, action='store_false', help=help_msg)
        else:
            parser.add_argument(
                arg_name,
                type=arg_type,
                default=default_value,
                help=help_msg
            )
    
    return parser.parse_args()

# 第二步：自动生成参数解析并更新变量值

args = auto_generate_args()
# 第三步：将解析结果更新回大写变量（保持变量名一致）
for var_name in [name for name in globals() if name.isupper() and not name.startswith('_')]:
    arg_name = var_name.lower()
    if hasattr(args, arg_name):
        globals()[var_name] = getattr(args, arg_name)
        
        

import torch 
import datetime
DEVICE = torch.device("cuda:" + str(GPU_ID) if torch.cuda.is_available() else "cpu")
EXPERIMENT_START_DATE = datetime.datetime.strptime(EXPERIMENT_START_DATE, '%Y-%m-%d')
EXPERIMENT_END_DATE = datetime.datetime.strptime(EXPERIMENT_END_DATE, '%Y-%m-%d')
