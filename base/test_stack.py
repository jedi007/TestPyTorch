import torch
 
#  函数中的输入inputs只允许是序列；且序列内部的张量元素，必须shape相等
# ----举例：[tensor_1, tensor_2,..]或者(tensor_1, tensor_2,..)，且必须tensor_1.shape == tensor_2.shape

# dim是选择生成的维度，必须满足0<=dim<len(outputs)；len(outputs)是输出后的tensor的维度大小

# 假设是时间步T1
T1 = torch.tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
# 假设是时间步T2
T2 = torch.tensor([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])


print(torch.stack((T1,T2),dim=0).shape)
print(torch.stack((T1,T2),dim=1).shape)
print(torch.stack((T1,T2),dim=2).shape)
# print(torch.stack((T1,T2),dim=3).shape) #error
# outputs:
# torch.Size([2, 3, 3])
# torch.Size([3, 2, 3])
# torch.Size([3, 3, 2])
# '选择的dim>len(outputs)，所以报错'
# IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)