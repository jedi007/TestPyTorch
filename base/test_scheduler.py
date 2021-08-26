def lr_scheduler_CosineAnnealingLR():
    import torch
    from torch import optim
    from torchvision.models import vgg11
    import matplotlib.pyplot as plt
 
    import numpy as np
 
    lr_list = []
    model = vgg11()
    LR = 0.01
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
    for epoch in range(100):
        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    plt.plot(range(100), lr_list, color='r')
    plt.show()

lr_scheduler_CosineAnnealingLR()