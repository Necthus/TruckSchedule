import os
import math
# 随机性参数
SEED = 2333
MAGIC_MAX_NUM  = 233333

# 运行设备参数
GPU_ID = 0

# 方法参数

DISPATCH_METHOD = 'Fastest' # 派单方法 可选项：'fastest'（优先派送最快的车） 'random'（随机派送） 'distance'（优先派送距离最近的车）
REPOSITION_METHOD = 'Retrace' # 车辆调度方法 可选项：'Urgent'（优先调度最紧急的车） 'Retrace'（优先调度回到原站点的车）

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

EXPERIMENT_START_DATE = '2024-05-01'  # 实验开始日期
EXPERIMENT_END_DATE = '2024-05-15'    # 实验结束日期



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