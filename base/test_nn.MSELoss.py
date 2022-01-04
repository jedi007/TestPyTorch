# 均方损失函数：
#     loss(xi,yi)=(xi−yi)^2

# (1)如果 reduce = False，那么 size_average 参数失效，直接返回向量形式的 loss
# (2)如果 reduce = True，那么 loss 返回的是标量
#     a)如果 size_average = True，返回 loss.mean();
#     b)如果 size_average = False，返回 loss.sum();

# 注意：默认情况下， reduce = True，size_average = True

import torch
import numpy as np

loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

a=np.array([[1,2],[3,4]])
b=np.array([[2,3],[4,5]])

input = torch.autograd.Variable(torch.from_numpy(a))
target = torch.autograd.Variable(torch.from_numpy(b))

loss = loss_fn(input.float(), target.float())

print(loss)
# tensor([[ 1.,  1.],
#         [ 1.,  1.]])


loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

a=np.array([[1,2],[3,4]])
b=np.array([[2,3],[4,4]])

input = torch.autograd.Variable(torch.from_numpy(a))
target = torch.autograd.Variable(torch.from_numpy(b))

loss = loss_fn(input.float(), target.float())

print(loss)
# tensor(0.7500)
# = (a - b)^2.mean()
print( torch.pow((a-b),2).mean() )
