
from parameter import *
import math



def coordinate_to_xy(lon, lat):
    
    x = (lon - ENV_MIN_LON) * KM_PER_DEGREE_LON / GRID_SIZE_KM
    y = (lat - ENV_MIN_LAT) * KM_PER_DEGREE_LAT / GRID_SIZE_KM
    
    return x, y
    
    



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




def get_relative_cos_sin_distance(A, B):
    """
    计算单个点B相对于A的cos、sin和距离（平面直角坐标）
    :param A: 位置A的坐标，格式为 (Ax, Ay)（元组/列表，数值类型）
    :param B: 位置B的坐标，格式为 (Bx, By)（元组/列表，数值类型）
    :return: cos_theta (float), sin_theta (float), distance (float)
    """
    # 提取单个点坐标
    Ax, Ay = A
    Bx, By = B
    
    # 计算x、y轴偏移量
    dx = Bx - Ax
    dy = By - Ay
    
    # 计算距离（勾股定理）
    distance = math.hypot(dx, dy)  # 等价于sqrt(dx²+dy²)，内置函数更高效
    
    # 处理A、B重合（避免除零）
    if distance < 1e-9:
        return 0.0, 0.0, 0.0
    
    # 计算cos和sin（直接用偏移量/距离，精准无三角函数误差）
    cos_theta = dx / distance
    sin_theta = dy / distance
    
    return cos_theta, sin_theta, distance



def get_next_move(current_position, target_position):
    """
    计算从当前位置到目标位置的下一步移动方向和距离

    参数:
    current_position (tuple): 当前位置坐标 (x1, y1)
    target_position (tuple): 目标位置坐标 (x2, y2)

    返回:
    move_direction (str): 移动方向，可能的值为"右"、"左"、"上"、"下"、"不动"
    move_distance (int/float): 移动距离
    """
    x1, y1 = current_position
    x2, y2 = target_position
    dx = x2 - x1
    dy = y2 - y1

    if dx != 0 and dy != 0:
        # 若x和y方向都有差值，可根据具体需求选择先移动哪个方向
        # 这里采用了哪个方向差值多优先哪个方向
        
        
        
        if TRACK_METHOD == 'diagonal':
        
            if abs(dx) > abs(dy):
                move_direction = ACTION_RIGHT if dx > 0 else ACTION_LEFT
                move_distance = abs(dx)
            else:
                move_direction = ACTION_UP if dy > 0 else ACTION_DOWN
                move_distance = abs(dy)
                
        elif TRACK_METHOD == 'x-first':
            move_direction = ACTION_RIGHT if dx > 0 else ACTION_LEFT
            move_distance = abs(dx)
        
        elif TRACK_METHOD == 'y-first':
            move_direction = ACTION_UP if dy > 0 else ACTION_DOWN
            move_distance = abs(dy)
            
            
            
            
            
    elif dx != 0:
        move_direction = ACTION_RIGHT if dx > 0 else ACTION_LEFT
        move_distance = abs(dx)
    elif dy != 0:
        move_direction = ACTION_UP if dy > 0 else ACTION_DOWN
        move_distance = abs(dy)
    else:
        move_direction = ACTION_STAY
        move_distance = 0

    return move_direction, move_distance




def minkowski_distance(point1, point2, p=2):
    """
    计算两点之间的Minkowski距离

    参数:
    point1 (tuple): 第一个点的坐标 (x1, y1)
    point2 (tuple): 第二个点的坐标 (x2, y2)
    p (int/float): Minkowski距离的阶数，默认为2（即欧几里得距离）

    返回:
    distance (float): 两点之间的Minkowski距离
    """
    return sum(abs(a - b) ** p for a, b in zip(point1, point2)) ** (1 / p)