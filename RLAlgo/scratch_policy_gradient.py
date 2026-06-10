import os

import torch

from network.base_net import StationScoringNet
from parameter import DEVICE, RL_GAMMA, RL_LR
from toolkit.log_analysis import find_latest_checkpoint
from toolkit.time import print_with_time


class ScratchPolicyGradientReposition:
    """Policy-gradient reposition learner trained from random initialization."""

    def __init__(self, input_dim=12, hidden_dims=(128, 64), lr=RL_LR, gamma=RL_GAMMA,
                 entropy_coef=0.02, device=DEVICE):
        self.policy_net = StationScoringNet(input_dim, hidden_dims, normalize=False).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device
        self.reset_transition()

    def reset_transition(self):
        self.transition = {
            'states': [],
            'actions': [],
            'rewards': [],
        }

    def take_action(self, station_features, force_greedy=False):
        x = torch.tensor(station_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        scores = self.policy_net(x).squeeze(0)
        probs = torch.softmax(scores, dim=-1)

        if force_greedy:
            action = torch.argmax(probs)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        return action.item()

    def update(self, transition_dict=None):
        if transition_dict is None:
            transition_dict = self.transition

        rewards = transition_dict['rewards']
        states = transition_dict['states']
        actions = transition_dict['actions']

        if not rewards:
            return

        returns = []
        g = 0.0
        for reward in reversed(rewards):
            g = reward + self.gamma * g
            returns.insert(0, g)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        losses = []
        entropies = []
        self.optimizer.zero_grad()

        for state, action_idx, advantage in zip(states, actions, returns):
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            scores = self.policy_net(x).squeeze(0)
            probs = torch.softmax(scores, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = torch.tensor(action_idx, dtype=torch.long).to(self.device)
            losses.append(-dist.log_prob(action) * advantage)
            entropies.append(dist.entropy())

        loss = torch.stack(losses).sum()
        if entropies:
            loss = loss - self.entropy_coef * torch.stack(entropies).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

    def save(self, save_dir, episode_num):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'checkpoint_epoch{episode_num}.pt')
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, save_dir, episode_num):
        path = os.path.join(save_dir, f'checkpoint_epoch{episode_num}.pt')
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print_with_time(f"从{path}加载Scratch Reposition模型")

    def model_initialization(self, save_dir, load_episode=-1):
        if load_episode >= 0:
            self.load(save_dir, load_episode)
            print_with_time(f'使用指定Scratch Reposition模型轮次：{load_episode}')
            return load_episode + 1

        last_epoch = find_latest_checkpoint(save_dir)
        if last_epoch == -1:
            print_with_time(f'没有在{save_dir}找到Scratch Reposition模型参数，从随机初始化开始训练')
            return 0

        self.load(save_dir, last_epoch)
        print_with_time(
            f'从{save_dir}加载Scratch Reposition模型参数，最后训练轮次为{last_epoch}，'
            f'本次从第{last_epoch + 1}轮开始训练'
        )
        return last_epoch + 1
