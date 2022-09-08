from env import *
from itertools import count

env = ENV(15)

env.show()

player = env.player
for t in count():
    action = input("第{}步， 请输入落子位置: ".format(t + 1))
    actions = action.split(",")
    if len(actions) != 2:
        print("输入错误")
        continue
    row, col = int(actions[0]), int(actions[1])

    state, reward, player, info = env.step(player, row - 1, col -1)

    env.show()

    if env.done:
        color = "黑棋" if env.player == 1 else "白棋"
        print( color + "获胜！")
        break

