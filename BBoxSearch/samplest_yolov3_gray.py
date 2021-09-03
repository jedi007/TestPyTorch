import torch
from torch import tensor
import torch.nn as nn

class Convolutional(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,useBN = False):
        super(Convolutional, self).__init__()

        # 根据解析的网络结构一层一层去搭建
        self.module_list = nn.ModuleList()
        
        if isinstance(kernel_size, int):
            modules = nn.Sequential()
            modules.add_module("Conv2d", nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=kernel_size // 2 ,
                                                    bias=False))
            nn.init.kaiming_normal_(modules[0].weight.data)

            if useBN:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(out_channels))
            modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            self.module_list.append(modules)
        else:
            pass
    
    def forward(self, x, verbose=False):
        for i, module in enumerate(self.module_list):
            x = module(x)
        return x


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2):
        super(Residual, self).__init__()

        # 根据解析的网络结构一层一层去搭建
        self.module_list = nn.ModuleList()
        
        self.module_list.append( Convolutional(in_channels,out_channels1,1,1) )
        self.module_list.append( Convolutional(out_channels1,out_channels2,3,1) )

    def forward(self, x, verbose=False):
        x1 = self.module_list[0](x)
        x1 = self.module_list[1](x1)

        return x+x1

class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x, verbose=False):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)

        return torch.cat( (x3,x2,x1,x),1 )

class YOLOv3Model_Gray(nn.Module):
    """
    YOLOv3 spp object detection model
    """
    def __init__(self,  verbose=True):
        super(YOLOv3Model_Gray, self).__init__()
        
        # 根据解析的网络结构一层一层去搭建
        self.module_list = nn.ModuleList()

        self.module_list.append( Convolutional(1,32,3,1) )
        self.module_list.append( Convolutional(32,64,3,2) )

        self.module_list.append( Convolutional(64,128,3,1) )
        self.module_list.append( Convolutional(128,128,3,2, useBN = True) )

        #self.module_list.append( Convolutional(128,128,3,1) )

        #Residual x8
        # self.module_list.append( Residual(128,64,128) )
        # self.module_list.append( Residual(128,64,128) )
        # self.module_list.append( Residual(128,64,128) )
        # self.module_list.append( Residual(128,64,128) )
        # self.module_list.append( Residual(128,64,128) )
        # self.module_list.append( Residual(128,64,128) )
        # self.module_list.append( Residual(128,64,128) )
        # self.module_list.append( Residual(128,64,128) )

        self.module_list.append( Convolutional(128,256,3,1) )
        self.module_list.append( Convolutional(256,128,3,2) )

        

        self.module_list.append( Convolutional(128,128,3,1) )
        self.module_list.append( Convolutional(128,128,3,2, useBN = True) )

        
        self.module_list.append( Convolutional(128,128,3,1 ) )
        self.module_list.append( Convolutional(128,128,3,1) )
        self.module_list.append( Convolutional(128,128,3,1) )
        self.module_list.append( Convolutional(128,128,3,1, useBN = True) )
        self.module_list.append( Convolutional(128,128,3,1) )
        self.module_list.append( Convolutional(128,128,3,1) )
        self.module_list.append( Convolutional(128,64,3,1, useBN = True) )



        
        
        modules = nn.Sequential()
        modules.add_module("Conv2d", nn.Conv2d(in_channels=64,
                                                out_channels=1,
                                                kernel_size=1,
                                                stride=1,
                                                padding=1 // 2 ,
                                                bias=True))
        self.module_list.append(modules)




        # 打印下模型的信息，如果verbose为True则打印详细信息
        #self.info(verbose)

    def forward(self, x, verbose=False):
        # yolo_out收集每个yolo_layer层的输出
        # out收集每个模块的输出
        if verbose:
            print('in x: ', x.shape)
            str = ""

        # for i in range( len(self.module_list) ):
        #     name = self.module_list[i].__class__.__name__
        #     print(name)
        for i in range( len(self.module_list) ):
            x = self.module_list[i](x)

        #print("out x.shape: ",x.shape)

        batchsize = x.shape[0]
        gradsize = x.shape[2]
        out = x.view(batchsize, gradsize, gradsize)

        torch.sigmoid_( out )
        return out

    def info(self, verbose=True):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        model_info(self, verbose)

    def loadPublicPt(self, weightsfile, device):
        local_dic = self.state_dict()
        load_dic = torch.load(weightsfile, map_location=device)
        newdic = dict( zip(local_dic.keys(),load_dic["model"].values()) )

        #print("newdic: ",newdic)

        self.load_state_dict( newdic )
        print("load model successed")

    
    

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

    fs = ''
    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


if __name__ == '__main__':
    model = YOLOv3Model_Gray()
    device = torch.device("cuda:0")


    model.to(device)
    model.train()


    img_size = 512
    input_size = (img_size, img_size)

    img = torch.ones((1, 1, img_size, img_size), device=device)
    pred = model(img)

    print( "pred.shape: ", pred.shape )
    print( "pred[0][0]: ", pred[0][0] )