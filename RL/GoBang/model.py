import torch.nn as nn
import os
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 16, 3)


        self.fc1 = nn.Linear(16*11*11, 32)

        self.action_head = nn.Linear(32, action_space)
        self.value_head = nn.Linear(32, 1) # Scalar Value

        self.save_actions = []
        self.rewards = []

        os.makedirs('./GoBang_Model', exist_ok=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1, 16*11*11)

        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value