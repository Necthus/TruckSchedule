import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from model import *

class REINFORCE:
    def __init__(self, feature_dim, hidden_dim, learning_rate=1e-3, gamma=0.98,
                 device='cuda:0', params_path=''):
        self.scorenet = ScoreNetwork(feature_dim,hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.scorenet.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device
        
        if os.path.exists(params_path):
            self.scorenet.load_state_dict(torch.load(params_path))
            print("Load model parameters from {}".format(params_path))
        else:
            print("No model parameters found at {}".format(params_path))

    def take_action(self, features):  # 根据动作概率分布随机采样
        
        num_actions =  len(features)
        
        state = torch.tensor(features, dtype=torch.float).to(self.device)
        
        probs = self.scorenet(state)
        probs = probs.cpu().detach().numpy()
        action = np.random.choice(num_actions,p=probs)
        return action
    
    def take_action_max_prob(self, features):  # 根据最大动作概率分布随机采样
        
        num_actions =  len(features)
        
        state = torch.tensor(features, dtype=torch.float).to(self.device)
        
        probs = self.scorenet(state)
        probs = probs.cpu().detach().numpy()
        max_index = probs.argmax()
        return max_index
    

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor(state_list[i],dtype=torch.float).to(self.device)
            action = torch.tensor(action_list[i]).to(self.device)
            log_prob = torch.log(self.scorenet(state)[action])
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降