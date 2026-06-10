import torch
from torch import nn


class MLP(nn.Module):
    """Simple MLP model"""

    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class StationScoringNet(nn.Module):
    """为每个厂站打分，支持内部标准化（用于预训练权重与RL推理兼容）"""

    def __init__(self, input_dim=8, hidden_dims=(64, 32), normalize=False):
        super().__init__()
        self.normalize = normalize
        if normalize:
            self.register_buffer('feat_mean', torch.zeros(input_dim))
            self.register_buffer('feat_std', torch.ones(input_dim))
        self.mlp = MLP(input_dim, hidden_dims, 1)

    def forward(self, station_features):
        """station_features: (batch, num_stations, input_dim) -> (batch, num_stations)"""
        if self.normalize:
            station_features = (station_features - self.feat_mean) / self.feat_std
        scores = self.mlp(station_features).squeeze(-1)
        return scores

    def set_normalization(self, mean, std):
        """设置归一化参数"""
        self.feat_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.tensor(std, dtype=torch.float32))
