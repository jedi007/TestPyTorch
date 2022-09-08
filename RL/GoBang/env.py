from cgitb import reset
import torch
from typing import Tuple

class ENV():
    def __init__(self, block_size):
        self.block_size = block_size

        self.reset()
    
    def reset(self):
        self.board = torch.zeros((self.block_size, self.block_size))
        self.player = 1
        self.done = False
        return self.board.clone()

    def show(self):
        s = " A\t"
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
            return self.board.clone(), 0, self.player, False 
        if player != self.player:
            return self.board.clone(), 0, self.player, False
        if self.board[row][col] != 0:
            print("return -10")
            self.done = True
            return self.board.clone(), -10, self.player, True

        if player == 1:
            self.board[row][col] = 1
        elif player == 0:
            self.board[row][col] = 0.5

        self.checkwin(row, col)
        reward = 1 if self.done else 0
        if not self.done:
            self.player = (player + 1) % 2  
        
        if render:
            self.show()
        return self.board.clone(), reward, self.player, True
    
    def checkwin(self, row: int, col: int):
        color = self.board[row][col]
        count_l_r = 1
        count_u_d = 1
        count_lu_rd = 1
        count_ru_ld = 1
        for i in range(1,5):
            if col - i >= 0:
                if self.board[row][col - i] == color:
                    count_l_r += 1
            if col + i < self.block_size:
                if self.board[row][col + i] == color:
                    count_l_r += 1   

            if row - i >= 0:
                if self.board[row - i][col] == color:
                    count_u_d += 1
            if row + i < self.block_size:
                if self.board[row + i][col] == color:
                    count_u_d += 1

            if col - i >= 0 and row - i >= 0:
                if self.board[row - i][col - i] == color:
                    count_lu_rd += 1
            if col + i < self.block_size and row + i < self.block_size:
                if self.board[row + i][col + i] == color:
                    count_lu_rd += 1

            if row - i >= 0 and col + i < self.block_size:
                if self.board[row - i][col + i] == color:
                    count_ru_ld += 1   
            if row + i < self.block_size and col - i >= 0:
                if self.board[row + i][col - i] == color:
                    count_ru_ld += 1  

        self.done = count_l_r >= 5 or  count_u_d >= 5 or count_lu_rd >= 5 or count_ru_ld >= 5
        

