import torch
import torch.nn as nn
import torch.nn.functional as F

def myCellOption(input, hidden, weight_ih, bias_ih, weight_hh, bias_hh ):
    r_i = torch.mm(input, weight_ih.transpose(0, 1)) + bias_ih    #效果类似于input 经过 神经元数量为 hidden_size 的全连接层
    print("r_i.shape: ", r_i.shape)

    r_h = torch.mm(hidden, weight_hh.transpose(0, 1)) + bias_hh
    print("r_h.shape: ", r_h.shape)

    res = torch.tanh(r_i+r_h)
    print("my res: ", res)

    return res


input_size = 4  # input_size means the lengths of one-hot encode vector, for example, the code [... 128 dim ...] of 'o' in "hello"
batch_size = 1
seq_len = 5    # it means the length of the whole sequence  rather than one-hot encode vector

hidden_size = 5

# the vector dimension of input and output for every sample x
Cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

print("Cell.weight_ih.shape: ", Cell.weight_ih.shape)
print("Cell.bias_ih.shape: ", Cell.bias_ih.shape)
print("Cell.weight_ih: ", Cell.weight_ih)
print("Cell.weight_hh.shape: ", Cell.weight_hh.shape)
print("Cell.bias_hh.shape: ", Cell.bias_hh.shape)

data = torch.randn(seq_len, batch_size, input_size)     # (3,2,4)
print(data)
hidden = torch.zeros(batch_size, hidden_size)   # (2,4)
print(hidden)

for idx, input in enumerate(data):
    print("=" * 20, idx, "=" * 20)
    res = myCellOption(input, hidden, Cell.weight_ih, Cell.bias_ih, Cell.weight_hh, Cell.bias_hh) # is the same as Cell(input, hidden) means right

    print("input shape:", input.shape)
    print(input)

    print(hidden)

    hidden = Cell(input, hidden)

    print("hidden shape:", hidden.shape)
    print(hidden)

