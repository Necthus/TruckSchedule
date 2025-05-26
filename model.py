import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_size):
        super(ScoreNetwork, self).__init__()
        self.sn = nn.Sequential(nn.Linear(feature_dim,hidden_size),nn.ReLU(),nn.Linear(hidden_size,1))

    def forward(self, features):
        # actions 的形状为 (num_actions, input_size)
        scores = self.sn(features)  # 计算每个动作的得分
        probabilities = F.softmax(scores, dim=0).view(-1)  # 通过 Softmax 转换为概率分布
        return probabilities
    
