import torch.nn as nn
import os
import torch.nn.functional as F
import torch


class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, 1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 1, 3, padding=1)

        # self.fc1 = nn.Linear(state_space, 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(1024, 512)

        self.action_head = nn.Linear(state_space, action_space)
        self.value_head = nn.Linear(state_space, 1)

        self.save_actions = []
        self.rewards = []
        self.state_space = state_space

        os.makedirs('./GoBang_Model', exist_ok=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = x.view(-1, self.state_space)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))

        action_score = self.action_head(x)
        state_value = self.value_head(x)

        action_score = F.softmax(action_score, dim=-1)

        return action_score, state_value