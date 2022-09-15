from cgitb import reset
import torch
from typing import Tuple

class ENV():
    def __init__(self, block_size, device):
        self.block_size = block_size
        self.device = device

        self.reset()
    
    def reset(self):
        self.board = torch.zeros((self.block_size, self.block_size))
        self.player = 1
        self.done = False
        self.winer = None
        return self.board.clone().to(self.device)

    def show(self):
        s = "\n\n\n A\t"
        for i in range(self.block_size):
            if i + 1 < 10:
                s += " "
            s += str(i + 1) + " "
        print(s)
        print("")
        for row in range(self.block_size):
            s = " " + str(row + 1) + "\t"
            for col in range(self.block_size):
                if self.board[row][col] == 1:
                    s += " ● "
                elif self.board[row][col] == 0.5:
                    s += " ○ "
                else:
                    s += " + "
            print(s)
    
    def step(self, player: int, row: int, col: int, render = False):
        if self.done:
            return self.board.clone().to(self.device), 0, self.player, False 
        if player != self.player:
            return self.board.clone().to(self.device), 0, self.player, False
        if self.board[row][col] != 0:
            self.done = True
            return self.board.clone().to(self.device), -10, self.player, True

        if player == 1:
            self.board[row][col] = 1
        elif player == 0:
            self.board[row][col] = 0.5

        self.checkwin(row, col)
        if self.done:
            self.winer = player

        reward = 2000 if self.done else 1
        if not self.done:
            self.player = (player + 1) % 2  
        
        if render:
            self.show()
        return self.board.clone().to(self.device), reward, self.player, True
    
    def checkwin(self, row: int, col: int):
        color = self.board[row][col]

        # 横向
        score = 1
        left = col
        while True:
            left -= 1
            if left >= 0 and self.board[row][left] == color:
                score += 1
            else:
                break
        right = col
        while True:
            right += 1
            if right < self.block_size and self.board[row][right] == color:
                score += 1
            else:
                break
        if score >=5:
            self.done = True
            return 
        
        # 纵向
        score = 1
        left = row
        while True:
            left -= 1
            if left >= 0 and self.board[left][col] == color:
                score += 1
            else:
                break
        right = row
        while True:
            right += 1
            if right < self.block_size and self.board[right][col] == color:
                score += 1
            else:
                break
        if score >=5:
            self.done = True
            return 

        # 左上==>右下
        score = 1
        r_left = row
        c_left = col
        while True:
            r_left -= 1
            c_left -= 1
            if r_left >= 0 and c_left >= 0 and self.board[r_left][c_left] == color:
                score += 1
            else:
                break
        r_right = row
        c_right = col
        while True:
            r_right += 1
            c_right += 1
            if r_right < self.block_size and c_right < self.block_size and self.board[r_right][c_right] == color:
                score += 1
            else:
                break
        if score >=5:
            self.done = True
            return 

        # 右上==>左下
        score = 1
        r_left = row
        c_left = col
        while True:
            r_left -= 1
            c_left += 1
            if r_left >= 0 and c_left < self.block_size and self.board[r_left][c_left] == color:
                score += 1
            else:
                break
        r_right = row
        c_right = col
        while True:
            r_right += 1
            c_right -= 1
            if r_right < self.block_size and c_right >= 0 and self.board[r_right][c_right] == color:
                score += 1
            else:
                break
        if score >=5:
            self.done = True
            return 
        

