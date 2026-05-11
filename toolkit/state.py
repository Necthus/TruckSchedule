import numpy as np

def subtract_self_features(X):
    """
    输入: X为N×D的矩阵（N辆加油车，D维特征）
    输出: N×(N-1)×D的矩阵，每个子矩阵为其他加油车特征减去当前车特征
    """
    N, D = X.shape
    
    # 步骤1: 复制矩阵为N×N×D，每个子矩阵都是原特征矩阵
    all_vehicles = np.tile(X, (N, 1, 1))
    
    # 步骤2: 生成另一个N×N×D矩阵，每个子矩阵是第i辆车的特征复制N次
    self_features = np.repeat(X[:, np.newaxis, :], N, axis=1)
    
    # 步骤3: 两个矩阵相减
    diff_matrix = all_vehicles - self_features
    
    # 步骤4: 对每个子矩阵，抽掉第i行（i=0到N-1）
    mask = ~np.eye(N, dtype=bool)  # 创建掩码排除对角线元素
    result = diff_matrix[mask].reshape(N, N-1, D)
    
    return result.astype(float)


def generate_machine_relative_location(machines_loc:np.ndarray,tankers_loc:np.ndarray):
    """
    输入: tankers_loc是N×D的矩阵（N辆加油车，D=2维特征）machines_loc是MxD的矩阵（M辆农机，D=2维特征）
    输出: N×农机数量=M×D的矩阵，每个子矩阵为农机特征减去当前车特征
    """
    n_machine = machines_loc.shape[0]
    n_tanker = tankers_loc.shape[0]
    
    # 步骤1: 复制矩阵为n_tanker×n_machine ×2，每个子矩阵都是原农机位置矩阵（复制）
    machines_loc_repeat = np.tile(machines_loc, (n_tanker, 1, 1))
    
    # 步骤2: 生成另一个n_tanker×n_machine ×2矩阵，每个子矩阵是第i辆车的特征复制n_machine次
    minus_tankers_loc_repeat = np.repeat(tankers_loc[:, np.newaxis, :], n_machine, axis=1)
    
    # 步骤3: 两个矩阵相减
    relative_loc = machines_loc_repeat-minus_tankers_loc_repeat

    
    return relative_loc.astype(float)


if __name__ == "__main__":
    # 示例验证
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [1, 1]
    ])

    M = np.array([
        [0, 0],
        [89, 8],
        [32, 3],
        [234, 54],
        [3131, 44]
    ])

    result = subtract_self_features(X)
    print("输入矩阵形状:", X.shape)
    print("输出矩阵形状:", result.shape)
    print("\n结果示例 (第1辆车对应的其他车特征差):")
    print(result)  # 应该显示车0和车2相对于车1的特征差

    relative_loc = generate_machine_relative_location(M,X)
    print("\n农机相对位置矩阵形状:", relative_loc.shape)
