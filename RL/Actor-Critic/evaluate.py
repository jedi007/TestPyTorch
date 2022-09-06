from model import *
import gym
import torch
from itertools import count

if __name__ == '__main__':
    #Parameters
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    model = Policy(state_space, action_space)
    model.load_state_dict(torch.load("AC_CartPole_Model/ModelTraing500Times.pt"))
    model.eval()

    state = env.reset()
    for t in count():
        state = torch.from_numpy(state).float()
        probs, state_value = model(state)
        action = torch.argmax(probs)
        
        state, reward, done, info, _ = env.step(action.cpu().squeeze(0).numpy())
        env.render()

        if done or t >= 1000:
            break
    
    print("t: ", t)
