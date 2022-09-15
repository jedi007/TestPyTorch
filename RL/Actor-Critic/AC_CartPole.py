import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from model import *

#Parameters
env = gym.make('CartPole-v0')
env = env.unwrapped  # <TimeLimit<CartPoleEnv>> 据说gym的多数环境都用TimeLimit（源码）包装了，以限制Epoch，就是step的次数限制，比如限定为200次。
                     # 用env.unwrapped可以得到原始的类，原始类想step多久就多久，不会200步后失败

env.seed(1)
torch.manual_seed(1)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n
print("state_space: ", state_space)
print("action_space: ", action_space)

#Hyperparameters
learning_rate = 0.01
gamma = 0.99
episodes = 500
render = False
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


model = Policy(state_space, action_space)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(steps)
    RunTime = len(steps)

    path = './AC_CartPole-v0/' + 'RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 200 == 0:
        plt.savefig(path)
    plt.pause(0.0000001)

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.save_actions.append(SavedAction(m.log_prob(action), state_value))

    return action.item()


def finish_episode():
    R = 0
    save_actions = model.save_actions
    policy_loss = []
    value_loss = []
    rewards = []

    for r in model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob , value), r in zip(save_actions, rewards):
        reward = r - value.item() # 这个value的意义是当前局面（state）的评分。但是在训练时还有另一层意义，这个值反应的是通过之前的学习，得到的该state下的评分。
                                  # 这个r是当前action得到的实际得分，如果 r - value 大于0，则说明这次尝试的这个action在该局面（state）下，获得了超过以往的分数，
                                  # 应该鼓励在实际对战中，遇到该局面(state)，提升进行该操作（action）的概率

        policy_loss.append(-log_prob * reward)  # reward 大于0，-log_prob函数曲线翻转，变成随x轴递减的曲线。此时对其求导，加上gradient. 
                                                # 相当于调整f(x)=log_prob(x)中的x增大。 并且reward越大，gradient也越大，调整幅度也就越大
                                                # 实际意义即是：该state下，执行该action的概率调大
                                                # 反之，reward 小于0时，在该state下，执行该action的概率调小
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.save_actions[:]

def main():
    live_time = []
    for i_episode in range(episodes):
        state = env.reset()
        for t in count():
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if render: env.render()
            model.rewards.append(reward)

            if done or t >= 1000:
                break

        live_time.append(t)
        plot(live_time)
        finish_episode()
    
    modelPath = './AC_CartPole_Model/ModelTraing'+str(episodes)+'Times.pt'
    torch.save(model.state_dict(), modelPath)

if __name__ == '__main__':
    main()