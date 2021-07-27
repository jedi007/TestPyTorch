from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)


print("\n\n\n=========================================================1")
x = x.new_ones(5, 3, dtype=torch.double)      
# new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    
# override dtype!
print(x)                                      
# result has the same size


print(x.size())

print("\n\n\n=========================================================2")
y = torch.rand(5, 3)
print(y) 
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)

print("\n\n\n=========================================================3")
print(x[:, 1])
print(x[0:2, 1:3]) #左闭右开区间

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

#如果你有一个元素 tensor ，使用 .item() 来获得这个 value 。
x = torch.randn(1)
print(x)
print(x.item())