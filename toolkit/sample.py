import numpy as np

def sample_gaussian_with_clipping(size=1, mean=0.5, std=0.1, min_val=0.4, max_val=0.6):
        """
        生成高斯分布样本并进行截断处理
        
        参数:
            size: 采样数量
            mean: 高斯分布均值
            std: 高斯分布标准差
            min_val: 最小值界限
            max_val: 最大值界限
        
        返回:
            截断后的采样结果
        """
        # 生成高斯分布随机数
        samples = np.random.normal(loc=mean, scale=std, size=size)
        
        # 截断处理：小于min_val的设为min_val，大于max_val的设为max_val
        clipped_samples = np.clip(samples, min_val, max_val)
        
        return clipped_samples