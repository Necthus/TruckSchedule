import torch
import numpy as np
from network.base_net import StationScoringNet
from parameter import RL_LR, RL_GAMMA, DEVICE, REPOSITION_TRAIN_MODE
from toolkit.log_analysis import find_latest_checkpoint
from toolkit.time import print_with_time


class REINFORCEReposition:
    """REINFORCE算法，用于Reposition厂站选择"""

    def __init__(self, input_dim=8, hidden_dims=(64, 32), lr=RL_LR, gamma=RL_GAMMA,
                 device=DEVICE):
        self.policy_net = StationScoringNet(input_dim, hidden_dims, normalize=True).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device

    def reset_transition(self):
        self.transition = {
            'states': [],
            'actions': [],
            'rewards': [],
        }

    def take_action(self, station_features, mask=None, force_greedy=False):
        """
        station_features: (num_stations, input_dim) numpy array
        mask: (num_stations,) bool array, True表示不可选
        force_greedy: 是否强制贪心选择
        返回: 选中的station索引
        """
        x = torch.tensor(station_features, dtype=torch.float).unsqueeze(0).to(self.device)
        scores = self.policy_net(x).squeeze(0)  # (num_stations,)

        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)
            scores = scores.masked_fill(mask_tensor, float('-inf'))

        probs = torch.softmax(scores, dim=-1)

        if force_greedy:
            action = torch.argmax(probs)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        return action.item()

    def take_action_with_prob(self, station_features):
        """返回动作和概率（训练用）"""
        x = torch.tensor(station_features, dtype=torch.float).unsqueeze(0).to(self.device)
        scores = self.policy_net(x).squeeze(0)
        probs = torch.softmax(scores, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), probs[action].item()

    def update(self, transition_dict=None):
        if transition_dict is None:
            transition_dict = self.transition

        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        if not reward_list:
            return

        G = 0
        returns = []
        for reward in reversed(reward_list):
            G = self.gamma * G + reward
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        losses = []
        self.optimizer.zero_grad()
        for i, G in enumerate(returns):
            s = state_list[i]
            x = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
            scores = self.policy_net(x).squeeze(0)
            probs = torch.softmax(scores, dim=-1)

            action = torch.tensor([action_list[i]], dtype=torch.long).to(self.device)
            log_prob = torch.log(probs.gather(0, action) + 1e-8)
            loss = -log_prob * G
            losses.append(loss)

        torch.stack(losses).sum().backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

    def save(self, save_dir, episode_num):
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'checkpoint_epoch{episode_num}.pt')
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, save_dir, episode_num):
        import os
        path = os.path.join(save_dir, f'checkpoint_epoch{episode_num}.pt')
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print_with_time(f"从{path}加载Reposition模型")

    def model_initialization(self, save_dir):
        last_epoch = find_latest_checkpoint(save_dir)
        if last_epoch == -1:
            print_with_time(f'没有在{save_dir}找到Reposition模型参数，从头开始训练')
            return 0
        else:
            self.load(save_dir, last_epoch)
            print_with_time(f'从{save_dir}加载Reposition模型参数，最后训练轮次为{last_epoch}，本次从第{last_epoch + 1}轮开始训练')
            return last_epoch + 1
