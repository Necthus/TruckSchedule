def get_key_position(step_reward_dict, target_key):
    # 1. 提取所有key并转为列表（过滤非数值key，避免排序报错）
    keys = [k for k in step_reward_dict.keys()]
    # 2. 排序：默认按数值升序（可改为 reverse=True 降序）
    sorted_keys = sorted(keys)
    # 4. 返回位置（索引+1 表示“第几位”，若需0开始索引可直接返回 index）
    return sorted_keys.index(target_key) 


def average_per_n_elements(lst, n, handle_remainder=True):
    """
    列表每n个元素取一次平均
    
    参数：
        lst: 输入的数值列表（需为int/float类型）
        n: 每n个元素为一组计算平均
        handle_remainder: 不足n个元素的尾部是否计算平均（True=计算，False=丢弃）
    
    返回：
        每组平均值组成的新列表
    """
    averages = []
    # 按步长n遍历列表，i为每组起始索引
    for i in range(0, len(lst), n):
        # 截取当前组（i到i+n的元素）
        group = lst[i:i+n]
        # 处理尾部不足n个元素的情况
        if not handle_remainder and len(group) < n:
            break
        # 计算平均值（避免空列表报错）
        if group:
            avg = sum(group) / len(group)
            averages.append(avg)
    return averages



import statistics

def calculate_mean_std(lst):
    
    # 计算均值和样本标准差
    mean_val = statistics.mean(lst)
    # 若列表只有1个元素，样本标准差无意义（返回0或提示均可）
    if len(lst) == 1:
        std_val = 0.0
        print("警告：列表仅1个元素，样本标准差无统计意义，返回0.0")
    else:
        std_val = statistics.stdev(lst)
    
    return mean_val, std_val