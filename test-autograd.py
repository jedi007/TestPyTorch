import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

print("=======================================")
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
print("=======================================")


out.backward() #out.backward() 等同于out.backward(torch.tensor(1.))。
print(x.grad) #打印梯度 d(out)/dx





print("\n\n=======================================1")

x = torch.randn(3, requires_grad=True)
print(x)
y = x * 2
print(y)

print( y.data.norm() ) #首先，它对张量y每个元素进行平方，然后对它们求和，最后取平方根。 这些操作计算就是所谓的L2或欧几里德范数 。

while y.data.norm() < 1000:
    y = y * 2

print(y)