from env import *
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

block_size = 15
model_black = Actor(state_space=block_size**2, action_space=block_size**2)
model_black.load_state_dict(torch.load("GoBang_Model/ModelTraing100000_black.pt"))
model_black.to(device=device)
model_black.eval()

env = ENV(block_size, device)
env.reset()
state, reward, player, info = env.step(env.player, int(block_size/2), int(block_size/2))
env.show()

def select_action(state, model):
    probs, state_value = model(state)

    # 保证和a相同的维度大小
    zero = torch.zeros((1,225), device=device)
    one = torch.ones((1,225), device=device)

    x0 = state.clone().view(1,225)
    a = torch.where(x0 > 0.4, one * 0.5, x0) # 已经落子的位置设置为0.5
    a = torch.where(a < 0.4, one, a) # 还可以落子的空位设置为1
    a = torch.where(a < 0.6, zero, a) # 已经落子的位置0

    probs = probs.clone() * a
    probs /= probs.sum()

    action = torch.argmax(probs)

    row = int(action.item() / block_size)
    col = int(action.item() % block_size)

    return row, col

def main():
    step = 2
    player = env.player
    while True:
        # 玩家落子
        action = input("第{}步， 请输入落子位置: ".format(step))
        actions = action.split(",")
        if len(actions) != 2:
            print("输入错误")
            continue
        row, col = int(actions[0]), int(actions[1])
        state, reward, player, info = env.step(player, row - 1, col -1, render = True)
        step += 1

        # 机器落子
        row, col = select_action(state.view(1, 1, block_size, block_size), model_black)
        state, reward_white, player, info = env.step(player, row, col, render = True)
        step += 1


        if env.done:
            color = "黑棋" if env.player == 1 else "白棋"
            print( color + "获胜！")
            break


if __name__ == '__main__':
    main()