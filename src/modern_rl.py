"""
Modern RL Architectures for Flappy Bird:
- ActorCriticNetwork: Used by PPO (Dual heads: Policy/Actor and Value/Critic)
- ActorOnlyNetwork: Used by GRPO (Single head: Policy/Actor only, no Critic)
"""
import math

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class ActorCriticNetwork(nn.Module):
    """
    PPO Architecture (Actor-Critic):
    Shares a common CNN backbone to extract spatial visual features from 4 stacked frames,
    then branches into:
    1. Actor Head (Policy): Outputs logits for discrete actions [idle, flap]
    2. Critic Head (Value): Outputs a scalar state-value estimate V(s)
    """
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()

        # Shared CNN Backbone (Identical to DQN in deep_q_network.py)
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))
        self.fc_shared = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))

        # 1. Actor Head: 2 action logits (Action 0 = Idle, Action 1 = Flap)
        self.actor_head = nn.Linear(512, 2)

        # 2. Critic Head: 1 scalar predicting expected return V(s)
        self.critic_head = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)
        # Actor head: small weights + bias so that P(flap) ~= 0.1 in every state at init
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        with torch.no_grad():
            self.actor_head.bias.copy_(torch.tensor([0.0, math.log(0.1 / 0.9)]))
        # Critic head: gain 1
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, state):
        """
        Args:
            state: Tensor with shape [Batch_Size, 4, 84, 84] (pixel values 0/255 from pre_processing)
        Returns:
            action_logits: Tensor [Batch_Size, 2]
            state_value: Tensor [Batch_Size, 1]
        """
        state = state / 255.0  # pre_processing outputs 0/255 -> scale to 0/1
        features = self.conv1(state)
        features = self.conv2(features)
        features = self.conv3(features)
        features = features.view(features.size(0), -1)
        shared_embedding = self.fc_shared(features)

        action_logits = self.actor_head(shared_embedding)
        state_value = self.critic_head(shared_embedding)

        return action_logits, state_value

    def get_action_and_value(self, state, action=None):
        """
        Helper method for PPO rollout and training:
        Evaluates current state, samples an action (or evaluates a past action),
        and returns log_prob, entropy, and value estimate.
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)


class ActorOnlyNetwork(nn.Module):
    """
    GRPO Architecture (Actor-Only):
    Discards the Critic Head completely to save VRAM and avoid value estimation noise.
    Only outputs action logits for the policy. Advantage is computed via Group Statistics.
    """
    def __init__(self):
        super(ActorOnlyNetwork, self).__init__()

        # CNN Backbone
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))
        self.fc_shared = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))

        # Actor Head only: 2 action logits
        self.actor_head = nn.Linear(512, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)
        # Actor head: small weights + bias so that P(flap) ~= 0.1 in every state at init
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        with torch.no_grad():
            self.actor_head.bias.copy_(torch.tensor([0.0, math.log(0.1 / 0.9)]))

    def forward(self, state):
        """
        Args:
            state: Tensor with shape [Batch_Size, 4, 84, 84] (pixel values 0/255 from pre_processing)
        Returns:
            action_logits: Tensor [Batch_Size, 2]
        """
        state = state / 255.0  # pre_processing outputs 0/255 -> scale to 0/1
        features = self.conv1(state)
        features = self.conv2(features)
        features = self.conv3(features)
        features = features.view(features.size(0), -1)
        shared_embedding = self.fc_shared(features)

        action_logits = self.actor_head(shared_embedding)
        return action_logits

    def get_action(self, state, action=None):
        """
        Helper method for GRPO group rollout:
        Samples an action from the policy distribution and returns action, log_prob, entropy.
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()
