from cmath import e
from env import *
from itertools import count
from model import *
from collections import namedtuple
from torch.distributions import Categorical
import torch.optim as optim


#Hyperparameters
learning_rate = 0.01
gamma = 0.99
episodes = 5
render = True
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

block_size = 15
env = ENV(block_size)

model_black = Actor(state_space=block_size**2, action_space=block_size**2)
model_white = Actor(state_space=block_size**2, action_space=block_size**2)
optimizer_black = optim.Adam(model_black.parameters(), lr=learning_rate)
optimizer_white = optim.Adam(model_white.parameters(), lr=learning_rate)

def finish_episode(model, optimizer):
    R = 0
    save_actions = model.save_actions
    policy_loss = []
    value_loss = []
    rewards = []

    for r in model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + e-6)

    for (log_prob , value), r in zip(save_actions, rewards):
        reward = r - value.item() 
        policy_loss.append(-log_prob * reward)  
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.save_actions[:]

def select_action(state, model):
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.save_actions.append(SavedAction(m.log_prob(action), state_value.view(1)))

    row = int(action.item() / block_size)
    col = action.item() % block_size
    return row, col

def main():
    for i_episode in range(episodes):
        state = env.reset()
        player = env.player
        for t in count():
            # 黑棋落子
            if env.board.sum().item() == 0:
                row = int(env.block_size/2)
                col = int(env.block_size/2)
            else:
                row,col = select_action(state.view(1, 1, block_size, block_size), model_black)
            state, reward_black, player, info = env.step(player, row, col, render)
            if env.done:
                model_black.rewards.append(reward_black)
                model_white.rewards.append(-1 if reward_black == 1 else 0)
                break

            # 白棋响应
            row, col = select_action(state.view(1, 1, block_size, block_size), model_white)
            state, reward_white, player, info = env.step(player, row, col, render)
            if env.done:
                model_black.rewards.append(-1 if reward_white == 1 else 0)
                model_white.rewards.append(reward_white)
                break

            model_black.rewards.append(reward_black)
            model_white.rewards.append(reward_white)

        finish_episode(model_black, optimizer_black)
        finish_episode(model_white, optimizer_white)
    
    modelPath = './GoBang_Model/ModelTraing'+str(episodes)
    torch.save(model_black.state_dict(), modelPath+'_black.pt')
    torch.save(model_white.state_dict(), modelPath+'_white.pt')

if __name__ == '__main__':
    main()