from __future__ import print_function
 
import torch
import torch.nn as nn
 
from PIL import Image
 
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt

from cocodataset import *

class ResBlock(nn.Module):

    def __init__(self,c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c,c,3,1,1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c),

        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.layer(x)+x)


class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2,1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128,64,3,1,1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,3,9,1,4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

def get_gram_matrix(f_map):
    n, c, h, w = f_map.shape
    f_map = f_map.reshape(n, c, h * w)
    #gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
    gram_matrix = torch.bmm(f_map, f_map.transpose(1, 2))
    return gram_matrix/ (c * h * w)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        #a = vgg16(True)
        a = torchvision.models.vgg16(pretrained=True)
        a = a.features
        self.layer1 = a[:4]
        self.layer2 = a[4:9]
        self.layer3 = a[9:16]
        self.layer4 = a[16:23]
	
    """输出四层的特征图"""
    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgsize = 256
loader = transforms.Compose([
    transforms.Scale([imgsize,imgsize]),# scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def load_image(image_name):
    image = cv2.imread(image_name)  # BGR
    image = cv2.resize(image, (imgsize, imgsize), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1)
    
    return image.unsqueeze(0)

#现在，让我们创建一个方法，通过重新将图片转换成PIL格式来展示，并使用plt.imshow展示它的拷贝。我们将尝试展示内容和风格图片来确保它们被正确的导入。
unloader = transforms.ToPILImage()  # reconvert into PIL image
plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(10000) # pause a bit so that plots are updated


image_style = load_image('D:/TestCode/TestPyTorch/style-transfer/style_1.jpg').cuda()

imshow(image_style)

print("image_style: ",image_style.shape)


vgg16 = VGG16().cuda()
t_net = TransNet().cuda()
# g_net.load_state_dict(torch.load('fst.pth'))





optimizer = torch.optim.Adam(t_net.parameters())
loss_func = nn.MSELoss().cuda()
data_set = COCODataSet("C:/COCO/images/val2017",imgsize)
batch_size = 1
data_loader = torch.utils.data.DataLoader(data_set, batch_size, True, drop_last=True)

"""计算分格,并计算gram矩阵"""
s1, s2, s3, s4 = vgg16(image_style)
s1 = get_gram_matrix(s1).detach().expand(batch_size,s1.shape[1],s1.shape[1])
s2 = get_gram_matrix(s2).detach().expand(batch_size,s2.shape[1],s2.shape[1])
s3 = get_gram_matrix(s3).detach().expand(batch_size,s3.shape[1],s3.shape[1])
s4 = get_gram_matrix(s4).detach().expand(batch_size,s4.shape[1],s4.shape[1])



start_epoch = 0
epochs = 10

for epoch in range(start_epoch, epochs):
    for step, image in enumerate(data_loader):
        """生成图片，计算损失"""
        image_c = image.cuda()
        image_t = t_net(image_c)
        out1, out2, out3, out4 = vgg16(image_t)
        # loss = loss_func(image_g, image_c)
        """计算风格损失"""
        loss_s1 = loss_func(get_gram_matrix(out1), s1)
        loss_s2 = loss_func(get_gram_matrix(out2), s2)
        loss_s3 = loss_func(get_gram_matrix(out3), s3)
        loss_s4 = loss_func(get_gram_matrix(out4), s4)
        loss_s = (loss_s1+loss_s2+loss_s3+loss_s4)*100000

        """计算内容损失"""
        c1, c2, c3, c4 = vgg16(image_c)

        # loss_c1 = loss_func(out1, c1.detach())
        # loss_c2 = loss_func(out2, c2.detach())
        # loss_c3 = loss_func(out3, c3.detach())
        loss_c = loss_func(out4, c4.detach())

        """总损失"""
        loss = loss_c + loss_s

        """清空梯度、计算梯度、更新参数"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print("epoch:{} =====> step:{}  loss_c:{}  loss_s:{}".format(epoch, step, loss_c.item(), loss_s.item()) )
    print("="*30,"\n"*5)
    torch.save(t_net.state_dict(), 'fst.pth')
    torchvision.utils.save_image([image_t[0], image_c[0]], f'D:/TestData/{epoch}.jpg', padding=0, normalize=True, range=(0, 1))
