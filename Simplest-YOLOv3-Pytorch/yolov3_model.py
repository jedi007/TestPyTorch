import torch
import torch.nn as nn

class YOLOModel(nn.Module):
    """
    YOLOv3 spp object detection model
    """
    def __init__(self, img_size=(416, 416), verbose=True):
        super(YOLOModel, self).__init__()
        self.input_size = img_size

        self.module_list = nn.ModuleList()
        # 统计哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是拼接)
        routs = []  # list of layers which rout to deeper layers

        self.addConvolutional(3, 32, 3, 1)
        self.addConvolutional(32, 64, 3, 2)

        #Residual X 1
        self.addConvolutional(64, 32, 1, 1)
        self.addConvolutional(32, 64, 1, 1)


    def forward(self, x, verbose=False):
        # yolo_out收集每个yolo_layer层的输出
        # out收集每个模块的输出

        print( "input x.shape: ",x.shape )

        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ""

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        print("out put yolo_out[0]: ",yolo_out[0].shape)
        print("out put yolo_out[1]: ",yolo_out[1].shape)
        print("out put yolo_out[2]: ",yolo_out[2].shape)

        if self.training:  # train
            return yolo_out
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p

    def info(self, verbose=True):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        model_info(self, verbose)

    def addConvolutional(self, in_channels,filters,ksize,stride):
        if isinstance(ksize, int):
            modules = nn.Sequential()
            modules.add_module("Conv2d", nn.Conv2d(    in_channels=in_channels,
                                                       out_channels=filters,
                                                       kernel_size=ksize,
                                                       stride=stride,
                                                       padding=ksize // 2,
                                                       bias=False))
            modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

            self.module_list.append(modules)
        else:
            raise TypeError("conv2d filter size must be int type.")

    


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 480, 640),), verbose=False)
        fs = ', %.1f GFLOPS' % (macs / 1E9 * 2)
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


if __name__ == '__main__':
    model = YOLOModel(img_size=(416, 416))
    model.info()