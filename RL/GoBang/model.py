import torch.nn as nn
import os
import torch.nn.functional as F
import torch


class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()

        # self.conv1 = nn.Conv2d(1, 32, 3)
        # self.conv2 = nn.Conv2d(32, 16, 3)

        self.fc1 = nn.Linear(state_space, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)

        self.action_head = nn.Sequential(nn.Linear(512, 512),
                                         nn.Linear(512, action_space))
        self.value_head = nn.Sequential(nn.Linear(512, 256),
                                        nn.Linear(256, 1))

        self.save_actions = []
        self.rewards = []
        self.state_space = state_space

        os.makedirs('./GoBang_Model', exist_ok=True)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))

        x = x.view(-1, self.state_space)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        action_score = self.action_head(x)
        state_value = self.value_head(x)

        action_score = F.softmax(action_score, dim=-1)

        return action_score, state_value