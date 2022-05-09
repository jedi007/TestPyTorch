import gym

env = gym.make("CartPole-v0")  ### 或者 env = gym.make("CartPole-v0").unwrapped 开启无锁定环境训练

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("state_size: ",state_size)
print("action_size: ",action_size)


state = env.reset()
print("state: ",state)