import torch
from torch import nn
import math
 

print("\n\n===================================================================nn.CrossEntropyLoss")
a=torch.Tensor([[4,8,3]])
y=torch.Tensor([2.]).long()
print(a.numpy())
print(y.numpy(),y.type())
criteon = nn.CrossEntropyLoss()   #nn.CrossEntropyLoss会自动加上Sofrmax层。
 
loss = criteon(a, y)
print("loss=",loss.item())
y1=torch.Tensor([1.]).long()
y2=torch.Tensor([0.]).long()
print("loss1=",criteon(a, y1).item())
print("loss2=",criteon(a, y2).item())

print("\n手动计算： ")
x = [4,8,3]
loss0 = -x[0]+math.log(math.exp(x[0])+math.exp(x[1])+math.exp(x[2]))
loss1 = -x[1]+math.log(math.exp(x[0])+math.exp(x[1])+math.exp(x[2]))
loss2 = -x[2]+math.log(math.exp(x[0])+math.exp(x[1])+math.exp(x[2]))
print("loss0: ",loss0)
print("loss1: ",loss1)
print("loss2: ",loss2)



print("\n\n===================================================================nn.BCELoss")
a=torch.Tensor([6,3,-4,6])
y=torch.Tensor([1,0,0,1])
print(a.numpy())
print(y.numpy(),y.type())
criteon = nn.BCELoss()   
 
pred=nn.Sigmoid()(a)
print("pred=",pred)
loss = criteon(pred, y)
print("loss=",loss.item())


print("\n手动计算： ")
label = [1,0,0,1]
r0 =  label[0] * math.log( pred[0] ) + (1-label[0]) *math.log(1 - pred[0])
r1 =  label[1] * math.log( pred[1] ) + (1-label[1]) *math.log(1 - pred[1])
r2 =  label[2] * math.log( pred[2] ) + (1-label[2]) *math.log(1 - pred[2])
r3 =  label[3] * math.log( pred[3] ) + (1-label[3]) *math.log(1 - pred[3])
print(r0)
print(r1)
print(r2)
print(r3)

print(type(r0))
loss=(-1/4)*(r0+r1+r2+r3)
print("loss: ",loss)