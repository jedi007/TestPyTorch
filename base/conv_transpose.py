import torch

seed = 0
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子


data = torch.randint(1,4,(1,3,2,2)).to(torch.float32)
print("data: ",data)

in_channels = 3
out_channels = 8


conv_t1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=1)
print("conv_t1.weight: ",conv_t1.weight.shape)
out1 = conv_t1(data)
print("out1.shape: ",out1.shape)   # input size: 2*2  周围填充k-p-1 = 2 - 0 - 1 size==>4*4  再用2*2的kernel做卷积 size==> 3*3


conv_t2 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
print("conv_t2.weight: ",conv_t2.weight.shape)
out2 = conv_t2(data)
print("out2.shape: ",out2.shape)# input size: 2*2  周围填充k-p-1 = 2 - 0 - 1;元素间填充s-1 = 1行、列 size==>5*5  再用2*2的kernel做卷积 size==> 4*4