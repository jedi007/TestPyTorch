import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import cv2



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ",device)

    model = torch.load("D:/TestData/mnist_fullmodel.pth")
    model.cuda().eval()

    print(model)

    img = cv2.imread('D:/TestData/index1.png',cv2.IMREAD_GRAYSCALE)

    img = img / 255

    print( img )

    print(img.shape)

    img_tensor = torch.tensor(img)


    print(img_tensor.shape)

    
    tensor_img = torch.tensor(img).view(1,1,28,28).float().cuda()

    print(tensor_img.shape)
    print( type(tensor_img) )

    model.eval()
    net = torch.jit.trace(model, tensor_img)
    net.save('D:/TestData/mnist_jit.pt')
    output=net(tensor_img)
    print(output)

    print("====================================begin")
    outputs = model( tensor_img )
    print("====================================end")
    
    print( outputs )

    r = torch.max(outputs.data, 1)
    print("r: ",r)


    # out = model( torch.tensor(img) )
    # print( out )
    # print( torch.max(out,1) )



