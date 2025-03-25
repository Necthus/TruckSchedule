import math

def calculate_range_dimensions(self_range):
    """
    计算给定经纬度范围的宽度和高度。
    
    Args:
        self_range (list): 格式为 [longitude_min, longitude_max, latitude_min, latitude_max]
        
    Returns:
        tuple: (width, height) 范围的宽度和高度
    """
    longitude_min, longitude_max, latitude_min, latitude_max = self_range

    # 计算范围的宽度
    width_km = 111 * math.cos(math.radians((latitude_min + latitude_max) / 2)) * (longitude_max - longitude_min)

    # 计算范围的高度
    height_km = 111 * (latitude_max - latitude_min)

    return width_km, height_km


import numpy as np

def softmax(x):
    e_x = np.exp(x)  
    return e_x / np.sum(e_x)

def normalized_reciprocal(x):
    recip = 1/(x+1)  
    return recip / np.sum(recip)