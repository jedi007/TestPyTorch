import torch
import torch.nn as nn

class BBoxSearchModel(nn.Module):
    def __init__(self, ):
        super(BBoxSearchModel, self).__init__()

        # 根据解析的网络结构一层一层去搭建
        self.module_list = nn.ModuleList()

        modules = nn.Sequential()
        
        modules.add_module("Conv2d", nn.Conv2d(in_channels=3,
                                        out_channels=32,
                                        kernel_size=3,
                                        stride=1,
                                        bias=False))

        modules.add_module("BatchNorm2d", nn.BatchNorm2d(32))
        modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

        self.module_list.append(modules)

    def forward(self, x, verbose=False):
        print( "input x.shape: ",x.shape )

        x = self.module_list[0](x)

        return x