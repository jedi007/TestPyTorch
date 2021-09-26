from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
from torch import Tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数
        else:
            # 当只有project_conv时的情况
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)  # 注意有激活函数

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result


class MyEfficientNetV2_S(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2):
        super(MyEfficientNetV2_S, self).__init__()
        
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        #stage 0
        self.stem = ConvBNAct(3,
                              24,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

        blocks = []
        #stage 1
        blocks.append(  FusedMBConv(kernel_size=3, input_c=24, out_c=24, expand_ratio=1, stride=1, se_ratio=0, drop_rate=0.0, norm_layer=norm_layer)  )     #0
        blocks.append(  FusedMBConv(kernel_size=3, input_c=24, out_c=24, expand_ratio=1, stride=1, se_ratio=0, drop_rate=0.005, norm_layer=norm_layer)  )   #1
        
        #stage 2
        blocks.append(  FusedMBConv(kernel_size=3, input_c=24, out_c=48, expand_ratio=4, stride=2, se_ratio=0, drop_rate=0.01, norm_layer=norm_layer)  )    #2
        blocks.append(  FusedMBConv(kernel_size=3, input_c=48, out_c=48, expand_ratio=4, stride=1, se_ratio=0, drop_rate=0.015, norm_layer=norm_layer)  )   #3
        blocks.append(  FusedMBConv(kernel_size=3, input_c=48, out_c=48, expand_ratio=4, stride=1, se_ratio=0, drop_rate=0.02, norm_layer=norm_layer)  )    #4
        blocks.append(  FusedMBConv(kernel_size=3, input_c=48, out_c=48, expand_ratio=4, stride=1, se_ratio=0, drop_rate=0.025, norm_layer=norm_layer)  )   #5

        #stage 3
        blocks.append(  FusedMBConv(kernel_size=3, input_c=48, out_c=64, expand_ratio=4, stride=2, se_ratio=0, drop_rate=0.03, norm_layer=norm_layer)  )    #6
        blocks.append(  FusedMBConv(kernel_size=3, input_c=64, out_c=64, expand_ratio=4, stride=1, se_ratio=0, drop_rate=0.035, norm_layer=norm_layer)  )   #7
        blocks.append(  FusedMBConv(kernel_size=3, input_c=64, out_c=64, expand_ratio=4, stride=1, se_ratio=0, drop_rate=0.04, norm_layer=norm_layer)  )    #8
        blocks.append(  FusedMBConv(kernel_size=3, input_c=64, out_c=64, expand_ratio=4, stride=1, se_ratio=0, drop_rate=0.045, norm_layer=norm_layer)  )   #9

        #stage 4
        blocks.append(  MBConv(kernel_size=3, input_c=64, out_c=128, expand_ratio=4, stride=2, se_ratio=0.25, drop_rate=0.05, norm_layer=norm_layer)  )     #10
        blocks.append(  MBConv(kernel_size=3, input_c=128, out_c=128, expand_ratio=4, stride=1, se_ratio=0.25, drop_rate=0.055, norm_layer=norm_layer)  )   #11
        blocks.append(  MBConv(kernel_size=3, input_c=128, out_c=128, expand_ratio=4, stride=1, se_ratio=0.25, drop_rate=0.06, norm_layer=norm_layer)  )    #12
        blocks.append(  MBConv(kernel_size=3, input_c=128, out_c=128, expand_ratio=4, stride=1, se_ratio=0.25, drop_rate=0.065, norm_layer=norm_layer)  )   #13
        blocks.append(  MBConv(kernel_size=3, input_c=128, out_c=128, expand_ratio=4, stride=1, se_ratio=0.25, drop_rate=0.07, norm_layer=norm_layer)  )    #14
        blocks.append(  MBConv(kernel_size=3, input_c=128, out_c=128, expand_ratio=4, stride=1, se_ratio=0.25, drop_rate=0.075, norm_layer=norm_layer)  )   #15

        #stage 5
        blocks.append(  MBConv(kernel_size=3, input_c=128, out_c=160, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.08, norm_layer=norm_layer)  )    #16
        blocks.append(  MBConv(kernel_size=3, input_c=160, out_c=160, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.085, norm_layer=norm_layer)  )   #17
        blocks.append(  MBConv(kernel_size=3, input_c=160, out_c=160, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.09, norm_layer=norm_layer)  )    #18
        blocks.append(  MBConv(kernel_size=3, input_c=160, out_c=160, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.095, norm_layer=norm_layer)  )   #19
        blocks.append(  MBConv(kernel_size=3, input_c=160, out_c=160, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.1, norm_layer=norm_layer)  )     #20
        blocks.append(  MBConv(kernel_size=3, input_c=160, out_c=160, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.105, norm_layer=norm_layer)  )   #21
        blocks.append(  MBConv(kernel_size=3, input_c=160, out_c=160, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.11, norm_layer=norm_layer)  )    #22
        blocks.append(  MBConv(kernel_size=3, input_c=160, out_c=160, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.115, norm_layer=norm_layer)  )   #23
        blocks.append(  MBConv(kernel_size=3, input_c=160, out_c=160, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.12, norm_layer=norm_layer)  )    #24

        #stage 6
        blocks.append(  MBConv(kernel_size=3, input_c=160, out_c=256, expand_ratio=6, stride=2, se_ratio=0.25, drop_rate=0.125, norm_layer=norm_layer)  )   #25
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.13, norm_layer=norm_layer)  )    #26
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.135, norm_layer=norm_layer)  )   #27
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.14, norm_layer=norm_layer)  )    #28
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.145, norm_layer=norm_layer)  )   #29
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.15, norm_layer=norm_layer)  )    #30
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.155, norm_layer=norm_layer)  )   #31
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.16, norm_layer=norm_layer)  )    #32
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.165, norm_layer=norm_layer)  )   #33
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.17, norm_layer=norm_layer)  )    #34
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.175, norm_layer=norm_layer)  )   #35
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.18, norm_layer=norm_layer)  )    #36
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.185, norm_layer=norm_layer)  )   #37
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.19, norm_layer=norm_layer)  )    #38
        blocks.append(  MBConv(kernel_size=3, input_c=256, out_c=256, expand_ratio=6, stride=1, se_ratio=0.25, drop_rate=0.195, norm_layer=norm_layer)  )   #39

        self.blocks = nn.Sequential(*blocks)

        
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(256,
                                               num_features,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        self.head = nn.Sequential(head)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x

    def model_info(self, verbose=False):
        # Plots a line-by-line description of a PyTorch model
        n_p = sum(x.numel() for x in self.parameters())  # number parameters
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)  # number gradients
        if verbose:
            print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
            for i, (name, p) in enumerate(self.named_parameters()):
                name = name.replace('module_list.', '')
                print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                    (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

        fs = ''
        print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(self.parameters())), n_p, n_g, fs))

if __name__ == '__main__':
    model = MyEfficientNetV2_S(num_classes=5)
    #print(model)

    model.model_info(True)