from __future__ import print_function
#%matplotlib inline
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os

current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录
print("current_work_dir: ",current_work_dir)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


        # nn.ConvTranspose2d init
        # in_channels: int,
        # out_channels: int,
        # kernel_size: _size_2_t,
        # stride: _size_2_t = 1,
        # padding: _size_2_t = 0,
        # output_padding: _size_2_t = 0,
        # groups: int = 1,
        # bias: bool = True,
        # dilation: int = 1,
        # padding_mode: str = 'zeros'
# 生成器代码
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是Z，进入卷积
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 1, 0, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
        x = input
        for i in range( len(self.main) ):
            print("i: ",i)
            x = self.main[i](x)
            print("x.shape: ",x.shape)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    # 为再现性设置随机seem
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # 如果你想要新的结果就是要这段代码
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    workers = 2
    batch_size = 64
    image_size = 28
    nc = 1
    nz = 100  #潜在向量 z 的大小(例如： 生成器输入的大小)
    ngf = 64 #生成器中特征图的大小

    ndf = 28 #判别器中的特征映射的大小
    num_epochs = 10
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1
    

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

    trainset = torchvision.datasets.MNIST(root='G:/AIData/pytorchData', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=False,drop_last = True)


    # 选择我们运行在上面的设备
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    
    
    
    netG = Generator(ngpu).to(device)

    #如果需要，管理multi-gpu
    if (device.type == 'cuda') and (ngpu > 1): netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.apply(weights_init)
    
    
    
    # 创建判别器
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # 应用weights_init函数随机初始化所有权重，mean= 0，stdev = 0.2
    netD.apply(weights_init)


    # 初始化BCELoss函数
    criterion = nn.BCELoss()

    # 创建一批潜在的向量，我们将用它来可视化生成器的进程
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # 在训练期间建立真假标签的惯例
    real_label = 1
    fake_label = 0

    # 为 G 和 D 设置 Adam 优化器
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



    # Training Loop

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # 对于数据加载器中的每个batch
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            real_imgs, labels = data
            real_imgs, labels = real_imgs.to(device), labels.to(device)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            label = torch.full((batch_size,), real_label, device=device)
            # Forward pass real batch through D

            output = netD(real_imgs).view(-1)
            # Calculate loss on all-real batch
            #exit(0)

            errD_real = criterion(output, label.float())
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()


            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)

            # Generate fake image batch with G
            fake = netG(noise)

            label.fill_(fake_label)

            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label.float())
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label.float())
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(trainloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        #保存/加载完整模型
        savename = "{}/savepath/save_Generator_model-{}.pt".format(current_work_dir, epoch)
        print("savename: ",savename)
        torch.save( netG.state_dict(), savename )
            
    

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



    


    # 从数据加载器中获取一批真实图像
    real_batch = next(iter(trainloader))

    # 绘制真实图像
    #imshow(real_batch[0][0])
    print(" real_batch[0]: ",real_batch[0].shape)
    imshow(torchvision.utils.make_grid(real_batch[0]))

    # 在最后一个epoch中绘制伪图像
    #imshow( fake[0] )

    netG = Generator(ngpu).to(device)
    netG.load_state_dict( torch.load("D:/pythonproject/test/savepath/save_Generator_model-9.pt") )

    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(noise).to( torch.device("cpu") ).detach() #to( torch.device("cpu") )


    imshow(torchvision.utils.make_grid(fake))


    
