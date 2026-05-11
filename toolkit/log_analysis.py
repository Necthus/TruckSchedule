import re

def smooth_list(l,smooth_range = 10):
    
    new_list = []
    list_len = len(l)
    
    if list_len<smooth_range:
        return []
    
    for i in range(smooth_range-1,list_len):
        
        smooth_item = sum(l[i-smooth_range+1:i+1])/smooth_range
        new_list.append(smooth_item)
        
        
    return new_list

def closest_multiple_of_20(epoch):
    base = (epoch // 20) * 20
    remainder = epoch % 20
    return base + 20 if remainder > 10 else base

def extract_values_from_file(file_path):
    """从文件中提取g = 后面的数值"""
    values = []
    pattern = re.compile(r'g\s*=\s*(\d+\.?\d*)')  # 匹配g = 数字（整数或浮点数）
    
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                value = float(match.group(1))  # 提取数字部分并转换为浮点数
                values.append(value)
    
    return values


import numpy as np
def find_best_epoch(file_path,smooth_range = 10,checkpoint_interval=20):
    
    rewards_list = extract_values_from_file(file_path)
    max_epoch = len(rewards_list)
    
    smooth_rewards_list = smooth_list(rewards_list,smooth_range)
    best_epoch = np.argmax(smooth_rewards_list)+smooth_range
    
    return closest_multiple_of_20(best_epoch)


def find_last_epoch(file_path,checkpoint_interval=20):
    rewards_list = extract_values_from_file(file_path)
    max_epoch = len(rewards_list)
    ret = closest_multiple_of_20(max_epoch)
    if ret >max_epoch:
        return ret-20
    else:
        return ret
    
    
    
import os
import re
import glob

def find_latest_checkpoint(directory,return_filename=True):
    """查找目录中数字最大的checkpoint_epoch{数字}.pt文件"""
    # 定义正则表达式模式，提取文件名中的数字
    pattern = r'checkpoint_epoch(\d+)\.pt'
    
    # 初始化最大数字和对应的文件名
    max_num = -1
    latest_file = None
    
    # 遍历目录中的所有文件
    for file_path in glob.glob(os.path.join(directory, 'checkpoint_epoch*.pt')):
        file_name = os.path.basename(file_path)
        match = re.search(pattern, file_name)
        
        if match:
            # 提取数字并转换为整数
            num = int(match.group(1))
            
            # 更新最大数字和对应的文件名
            if num > max_num:
                max_num = num
                latest_file = file_path
    
    return max_num
    

    
    
    
    