import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import  collections

env = gym.make('Pendulum-v1')
env.seed(2333)
torch.manual_seed(2333)    # 策略梯度算法方差很大，设置seed以保证复现性
env.reset()
env.render()
print('observation space:',env.observation_space)
print('action space:',env.action_space)
class ReplayBuffer():
    # 经验回放池
    def __init__(self):
        # 双向队列
        buffer_limit = 50000
        self.buffer = collections.deque(maxlen=buffer_limit)
        #通过 put(transition)方法 将最新的(𝑠, 𝑎, 𝑟, 𝑠′)数据存入 Deque 对象
    def put(self, transition):
        self.buffer.append(transition)
    #通过 sample(n)方法从 Deque 对象中随机采样出 n 个(𝑠, 𝑎, 𝑟, 𝑠′)数据
    def sample(self, n):
        # 从回放池采样n个5元组
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []
        # 按类别进行整理
        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
        # 转换成Tensor
        return torch.Tensor(s_lst), \
               torch.Tensor(a_lst), \
                      torch.Tensor(r_lst), \
                      torch.Tensor(s_prime_lst)


    def size(self):
        return len(self.buffer)


# 策略网络，也叫Actor网络，输入为state  输出为概率分布pi(a|s)
class Actor(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Actor, self).__init__()
        # self.linear  = nn.Linear(hidden_size, output_size)
        self.actor_net = nn.Sequential(
            nn.Linear(in_features=input_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=output_size)
        )
    def forward(self,state):
        x = self.actor_net(state)
        x = torch.tanh(x)
        return x

#值函数网络  输入是state，action输出是Q(s,a)
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.critic_net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, state,action):
        inputs = torch.cat([state,action],1)
        x = self.critic_net(inputs)
        return x


class DDPG():
    def __init__(self,state_size,action_size,hidden_size = 256,actor_lr = 0.001,ctitic_lr = 0.001,batch_size = 32):

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.actor_lr = actor_lr #actor网络学习率
        self.critic_lr = ctitic_lr#critic网络学习率
        # 策略网络，也叫Actor网络，输入为state  输出为概率分布pi(a|s)
        self.actor = Actor(self.state_size, self.hidden_size, self.action_size)
        #target actor网络 延迟更新
        self.actor_target = Actor(self.state_size, self.hidden_size, self.action_size)
        # 值函数网络  输入是state，action输出是Q(s,a)
        self.critic = Critic(self.state_size + self.action_size, self.hidden_size, self.action_size)
        self.critic_target = Critic(self.state_size + self.action_size, self.hidden_size, self.action_size)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []
        # 影子网络权值来自原网络，只不过延迟更新
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.gamma = 0.99
        self.batch_size = batch_size
        self.memory = ReplayBuffer()  # 创建回放池

        self.memory2 = []
        self.learn_step_counter = 0 #学习轮数 与影子网络的更新有关
        self.replace_target_iter = 200 #影子网络迭代多少轮更新一次
        self.cost_his_actor = []# 存储cost 准备画图
        self.cost_his_critic = []


    def choose_action(self,state):
        # 将state转化成tensor 并且维度转化为[3]->[1,3]  unsqueeze(0)在第0个维度上田间
        state = torch.Tensor(state).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        return action
    #critic网络的学习
    def critic_learn(self,s0,a0,r1,s1):
        #从actor_target通过状态获取对应的动作  detach()将tensor从计算图上剥离
        a1 = self.actor_target(s0).detach()
        #删减一个维度  [b,1,1]变成[b,1]
        a0 = a0.squeeze(2)
        y_pred = self.critic(s0,a0)
        y_target = r1 +self.gamma *self.critic_target(s1,a1).detach()
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        self.cost_his_critic.append(loss.item())
    #actor网络的学习
    def actor_learn(self,s0,a0,r1,s1):
        loss = -torch.mean(self.critic(s0, self.actor(s0)))
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        self.cost_his_actor.append(loss.item())
    #模型的训练
    def train(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
        #随机采样出 batch_size 个(𝑠, 𝑎, 𝑟, 𝑠′)数据
        s0, a0, r, s_prime = self.memory.sample(self.batch_size)
        self.critic_learn(s0, a0, r, s_prime)
        self.actor_learn(s0, a0, r, s_prime)

        self.soft_update(self.critic_target, self.critic, 0.02)
        self.soft_update(self.actor_target, self.actor, 0.02)
    #target网络的更新
    def soft_update(self,net_target, net, tau):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his_critic)), self.cost_his_critic)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



def main():
    print(env.observation_space.shape[0])
    print(env.action_space.shape[0])
    ddgp = DDPG(state_size=env.observation_space.shape[0],
                action_size=env.action_space.shape[0],
                hidden_size=256,
                actor_lr=0.001,
                ctitic_lr=  0.001,
                batch_size=32)

    print_interval = 4

    for episode in range(100):
        state = env.reset()
        episode_reward = 0

        for step in range(500):
            env.render()
            action0 = ddgp.choose_action(state)
            s_prime, r, done, info = env.step(action0)

            # 保存四元组
            ddgp.memory.put((state, action0, r, s_prime))
            episode_reward += r
            state = s_prime

            if done:  # 回合结束
                break

            if ddgp.memory.size() > 32:  # 缓冲池只有大于500就可以训练
                ddgp.train()

        if episode % print_interval == 0 and episode != 0:
            print("# of episode :{}, avg score : {:.1f}, buffer size : {}, "
                  .format(episode, episode_reward / print_interval, ddgp.memory.size()))
    env.close()
    ddgp.plot_cost()

if __name__ == "__main__":
    main()
