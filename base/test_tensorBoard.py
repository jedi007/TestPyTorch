# 查看tensorboard，控制台输入以下语句，--logdir=设置的日志文件夹。
# tensorboard --logdir=runs
# 打开浏览器输入http://localhost:6006

#或者在CMD中输入：C:\Users\lijie>tensorboard --logdir=D:\Study\GitHub\TestPyTorch\runs

import torch
import random
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs')

# 实例化模型
import torchvision.models as models
resnet18 = models.resnet18()
# alexnet = models.alexnet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# desnet = models.densenet161()
# inception =models.inception_v3()

model = resnet18
# 将模型写入tensorboard
init_img = torch.zeros((1, 3, 224, 224))
writer.add_graph(model, init_img)

tags = ["train_loss", "accuracy", "learning_rate"]
for epoch in range(30):
    mean_loss = random.randint(0,10)
    acc = random.random()
    learning_rate = 0.01*random.random()
    writer.add_scalar(tags[0], mean_loss, epoch)
    writer.add_scalar(tags[1], acc, epoch)
    writer.add_scalar(tags[2], learning_rate, epoch)


# add conv1 weights into tensorboard
writer.add_histogram(tag="conv1",
                        values=model.conv1.weight,
                        global_step=epoch)


# 绘制散点图
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(20200910)

fig, ax = plt.subplots() 
ax.plot(10*np.random.randn(100),10*np.random.randn(100),'o')
writer.add_figure(
    '林麻子matplotlib figure林祖泉', 
    fig, 
    global_step=None, 
    close=False, 
    walltime=None)




writer.close

