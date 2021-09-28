import torch
from torch import optim
from torchvision.models import vgg11
import matplotlib.pyplot as plt

import numpy as np
import math

def lr_scheduler_CosineAnnealingLR():
    lr_list = []
    model = vgg11()
    LR = 0.01
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
    for epoch in range(100):
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    plt.plot(range(100), lr_list, color='b')
    #plt.show()

def lr_scheduler_LambdaLR():
    epochs = 100

    lr_list = []
    model = vgg11()
    LR = 0.01
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1E-4)
    #lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - LR) + LR  # cosine  x 为 step,这里为0-100  lf得到的只是一个系数，再*LR才是实际中使用的lr
    lf = lambda x: 1+math.cos( (math.pi/10)*x ) #*0.5
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(epochs):
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    plt.plot(range(epochs), lr_list, color='r')
    plt.show()

lr_scheduler_CosineAnnealingLR()
lr_scheduler_LambdaLR()