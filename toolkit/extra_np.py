import numpy as np  


def custom_round(arr, upper_thresh=0.6, lower_thresh=0.4, mid_handle="keep"):
    """
    自定义numpy数组舍入函数
    :param arr: 输入numpy数组
    :param upper_thresh: 向上舍入阈值（>该值则进1），默认0.6
    :param lower_thresh: 向下舍入阈值（<该值则归x），默认0.4
    :param mid_handle: 中间区间（lower~upper）处理方式：
                       "keep"=保留原值，"round"=常规四舍五入，"up"=进1，"down"=归x
    :return: 舍入后的数组
    """
    # 分离整数部分和小数部分
    integer_part = np.floor(arr)  # 向下取整得到整数部分（如5.7→5，-2.3→-3）
    decimal_part = arr - integer_part  # 小数部分（始终≥0）
    
    # 创建结果数组（初始为整数部分）
    result = integer_part.copy()
    
    # 1. 大于上阈值：进1
    result[decimal_part > upper_thresh] += 1
    
    # 2. 小于下阈值：保持整数部分（已默认）
    
    return result