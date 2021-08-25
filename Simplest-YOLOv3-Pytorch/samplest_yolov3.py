import torch
from torch import tensor
import torch.nn as nn

class Convolutional(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
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

class YOLOBase(nn.Module):
    def __init__(self):
        super(YOLOBase, self).__init__()
    
    def forward(self, x, verbose=False):
        batchsize, _, ny, nx = x.shape

        x = self.module_list[0](x)
        x = self.module_list[1](x)

        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]
        # 原始x[batch, number anchor, xywh + obj + classes, grid, grid]
        x = x.view(batchsize, 3, 25, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction


        if self.training:
            return x
        else:
            device = x.device

            yv, xv = torch.meshgrid([torch.arange(ny, device=device),
                                     torch.arange(nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view(batchsize, 1, ny, nx, 2).float()

            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

            io = x.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            io[..., :4] *= self.stride  # 换算映射回原图尺度
            torch.sigmoid_(io[..., 4:])
            out = io.view(batchsize, -1, 25) # view [1, 3, 13, 13, 85] as [1, 507, 85]
            return out

class YOLOP0(YOLOBase):
    def __init__(self):
        super(YOLOP0, self).__init__()
        self.stride = 32

        #[ [10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326] ]
        self.anchors = torch.tensor([ [116,90],  [156,198],  [373,326] ])
        # 将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors / self.stride

        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, len(self.anchors), 1, 1, 2)

        # 根据解析的网络结构一层一层去搭建
        self.module_list = nn.ModuleList()
        
        self.module_list.append( Convolutional(512,1024,3,1) )

        modules = nn.Sequential()
        modules.add_module("Conv2d", nn.Conv2d(in_channels=1024,
                                                out_channels=75,
                                                kernel_size=1,
                                                stride=1,
                                                padding=1 // 2 ,
                                                bias=True))
        self.module_list.append(modules)

class YOLOP1(YOLOBase):
    def __init__(self):
        super(YOLOP1, self).__init__()

        self.stride = 16

        #[ [10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326] ]
        self.anchors = torch.tensor([ [30,61],  [62,45],  [59,119] ])
        # 将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors / self.stride

        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, len(self.anchors), 1, 1, 2)

        # 根据解析的网络结构一层一层去搭建
        self.module_list = nn.ModuleList()
        
        self.module_list.append( Convolutional(256,512,3,1) )

        modules = nn.Sequential()
        modules.add_module("Conv2d", nn.Conv2d(in_channels=512,
                                                out_channels=75,
                                                kernel_size=1,
                                                stride=1,
                                                padding=1 // 2 ,
                                                bias=True))
        self.module_list.append(modules)

class YOLOP2(YOLOBase):
    def __init__(self):
        super(YOLOP2, self).__init__()

        self.stride = 8

        #[ [10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326] ]
        self.anchors = torch.tensor([ [10,13],  [16,30],  [33,23] ])
        # 将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors / self.stride

        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, len(self.anchors), 1, 1, 2)

        # 根据解析的网络结构一层一层去搭建
        self.module_list = nn.ModuleList()
        
        self.module_list.append( Convolutional(128,256,3,1) )

        modules = nn.Sequential()
        modules.add_module("Conv2d", nn.Conv2d(in_channels=256,
                                                out_channels=75,
                                                kernel_size=1,
                                                stride=1,
                                                padding=1 // 2 ,
                                                bias=True))
        self.module_list.append(modules)

class YOLOv3Model(nn.Module):
    """
    YOLOv3 spp object detection model
    """
    def __init__(self,  verbose=True):
        super(YOLOv3Model, self).__init__()
        
        # 根据解析的网络结构一层一层去搭建
        self.module_list = nn.ModuleList()

        self.module_list.append( Convolutional(3,32,3,1) )
        self.module_list.append( Convolutional(32,64,3,2) )

        #Residual x1
        self.module_list.append( Residual(64,32,64) )

        self.module_list.append( Convolutional(64,128,3,2) )

        #Residual x2
        self.module_list.append( Residual(128,64,128) )
        self.module_list.append( Residual(128,64,128) )

        self.module_list.append( Convolutional(128,256,3,2) )

        #Residual x8
        self.module_list.append( Residual(256,128,256) )
        self.module_list.append( Residual(256,128,256) )
        self.module_list.append( Residual(256,128,256) )
        self.module_list.append( Residual(256,128,256) )
        self.module_list.append( Residual(256,128,256) )
        self.module_list.append( Residual(256,128,256) )
        self.module_list.append( Residual(256,128,256) )
        self.module_list.append( Residual(256,128,256) )
        #laylers 15 ================================================ for concatenate2

        self.module_list.append( Convolutional(256,512,3,2) )


        #Residual x8
        self.module_list.append( Residual(512,256,512) )
        self.module_list.append( Residual(512,256,512) )
        self.module_list.append( Residual(512,256,512) )
        self.module_list.append( Residual(512,256,512) )
        self.module_list.append( Residual(512,256,512) )
        self.module_list.append( Residual(512,256,512) )
        self.module_list.append( Residual(512,256,512) )
        self.module_list.append( Residual(512,256,512) )
        #laylers 24 ================================================ for concatenate1

        self.module_list.append( Convolutional(512,1024,3,2) )

        #Residual x4
        self.module_list.append( Residual(1024,512,1024) )
        self.module_list.append( Residual(1024,512,1024) )
        self.module_list.append( Residual(1024,512,1024) )
        self.module_list.append( Residual(1024,512,1024) )
        #laylers 29


        self.module_list.append( Convolutional(1024,512,1,1) )
        self.module_list.append( Convolutional(512,1024,3,1) )
        self.module_list.append( Convolutional(1024,512,1,1) )


        self.module_list.append( SPP() )

        self.module_list.append( Convolutional(2048,512,1,1) )
        self.module_list.append( Convolutional(512,1024,3,1) )
        self.module_list.append( Convolutional(1024,512,1,1) )
        #laylers 36 ================================================ for YOLOP0

        #YOLOP0
        self.module_list.append( YOLOP0() )


        self.module_list.append( Convolutional(512,256,1,1) )

        self.module_list.append( nn.Upsample(scale_factor=2) )

        #cat

        #Convolutional set1
        self.module_list.append( Convolutional(768,256,1,1) )
        self.module_list.append( Convolutional(256,512,3,1) )
        self.module_list.append( Convolutional(512,256,1,1) )
        self.module_list.append( Convolutional(256,512,3,1) )
        self.module_list.append( Convolutional(512,256,1,1) )
        #laylers 44 ================================================ for YOLOP1

        #YOLOP1
        self.module_list.append( YOLOP1() )
        


        self.module_list.append( Convolutional(256,128,1,1) )
        self.module_list.append( nn.Upsample(scale_factor=2) )
        #cat

        #Convolutional set2
        self.module_list.append( Convolutional(384,128,1,1) )
        self.module_list.append( Convolutional(128,256,3,1) )
        self.module_list.append( Convolutional(256,128,1,1) )
        self.module_list.append( Convolutional(128,256,3,1) )
        self.module_list.append( Convolutional(256,128,1,1) )
        #laylers 52 ================================================ for YOLOP0

        #YOLOP2
        self.module_list.append( YOLOP2() )



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
        
        for i in range(14):
            x = self.module_list[i](x)

        x = self.module_list[14](x)
        concatenate2 = x

        for i in range(15,23):
            x = self.module_list[i](x)

        x = self.module_list[23](x)
        concatenate1 = x

        for i in range(24,36):
            x = self.module_list[i](x)

        YOLOP0_branch = self.module_list[36](x)

        x = self.module_list[37](x)
        x = self.module_list[38](x)

        x = self.module_list[39]( torch.cat( (x,concatenate1),1 ) )

        for i in range(40,44):
            x = self.module_list[i](x)

        YOLOP1_branch = self.module_list[44](x)

        x = self.module_list[45](x)
        x = self.module_list[46](x)

        x = self.module_list[47]( torch.cat( (x,concatenate2),1 ) )

        for i in range(48,53):
            x = self.module_list[i](x)
        
        YOLOP2_branch = x

        if self.training:
            return [YOLOP0_branch,YOLOP1_branch,YOLOP2_branch]
            
        return torch.cat( (YOLOP0_branch,YOLOP1_branch,YOLOP2_branch),1 )

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
    model = YOLOv3Model()
    device = torch.device("cuda:0")

    model.loadPublicPt("D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/weights/yolov3spp-voc-512.pt",device)

    model.to(device)
    model.eval()


    img_size = 512
    input_size = (img_size, img_size)

    img = torch.ones((1, 3, img_size, img_size), device=device)
    pred = model(img)

    print( "pred.shape: ", pred.shape )
    print( "pred[0][0]: ", pred[0][0] )
    print( "pred[0][768]: ", pred[0][768] )
    print( "pred[0][3840]: ", pred[0][3840] )