from turtle import shape
import paddle.fluid.layers as F
import paddle.fluid.dygraph as nn
import paddle.fluid as fluid
import numpy as np
import paddle

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2D(in_planes, out_planes, filter_size=3, stride=stride, padding=1, bias_attr=False)

class ReLU(nn.Layer):
    def __init__(self):
        super(ReLU, self).__init__()
    def forward(self, x):
        return F.relu(x)

class Sigmoid(nn.Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()
    def forward(self, x):
        return F.sigmoid(x)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channel, out_channel,
                               filter_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm(out_channel)
        self.relu = ReLU()

        self.conv2 = nn.Conv2D(out_channel, out_channel,
                               filter_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, filter_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2D(planes, planes, filter_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, filter_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm(planes * 4)
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Layer):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, filter_size=7, stride=2, padding=3, bias_attr=False) # conv3x3(3, 64, stride=2) 
        self.bn1 = nn.BatchNorm(64)
        self.relu = ReLU()
        self.maxpool = nn.Pool2D(pool_padding=1,pool_size=3,pool_type="max",pool_stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = paddle.nn.AdaptiveAvgPool2D((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
  
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0]*m.weight.shape[1]*m.weight.shape[2]
                v = np.random.normal(loc=0.,scale=np.sqrt(2./n),size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

    def _make_layer(self, block, planes, block_num, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          filter_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, block_num):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x

    def model_info(self, verbose=False):
        # Plots a line-by-line description of a PyTorch model
        n_p = sum(x.numel() for x in self.parameters())  # number parameters
        print("n_p[0]: ",n_p[0])
        #return
        n_g = sum(x.numel() for x in self.parameters() if not x.stop_gradient)  # number gradients
        if verbose:
            print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
            for i, (name, p) in enumerate(self.named_parameters()):
                #name = name.replace('module_list.', '')
                print('%5g %40s %9s %12.10g %20s %10.3g %10.3g' %
                    (i, name, not p.stop_gradient, p.numel(), list(p.shape), p.mean(), p.std()))


        fs = ''
        print('Model Summary: %g layers, %12.10g parameters, %12.10g gradients%s' % (len(list(self.parameters())), n_p, n_g, fs))

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

if __name__ == "__main__":
    net = resnet18(5)
    # model = paddle.Model(net)
    # model.summary((-1, 3, 224, 224))
    # net.model_info(True)

    data = np.load(".\\resnet_18.npz", allow_pickle=True)
    print("data: ",type(data))

    #np.set_printoptions(suppress=True)
    print( "data.files", data.files)
    print( "type(data['resnet_dict']): ", type(data['resnet_dict']) )
    print( "type(data['resnet_dict']): ", type(data['resnet_dict']) )

    resnet18_dic = data['resnet_dict'].item()
    print("resnet18_dic: ",type(resnet18_dic))
    resnet18_parameters = [x for x in resnet18_dic.values()]
    print("resnet18_dic.values(): ", type(resnet18_dic.values()) )

    i = 0
    for m in resnet18_parameters:
        print("i,m: ", i, m.shape)
        i += 1


    iter = 0
    for m in net.sublayers():
        if isinstance(m, nn.Conv2D):
            #print("m.weight.shape(): ",m.weight.shape )
            #print("parameters.shape: ",list( resnet18_parameters[iter].shape ) )
            list2 = list( resnet18_parameters[iter].shape )
            assert m.weight.shape == list2, f"shape error 1 iter:{iter}"
            m.weight.set_value( resnet18_parameters[iter] )
            iter += 1
        elif isinstance(m, nn.BatchNorm):
            assert m.weight.shape == list(resnet18_parameters[iter].shape), f"shape error 2 iter:{iter}"
            m.weight.set_value( resnet18_parameters[iter] )
            iter += 1
            
            m.bias.set_value( resnet18_parameters[iter] )
            iter += 1

            m._mean.set_value( resnet18_parameters[iter] )
            iter += 1

            m._variance.set_value( resnet18_parameters[iter] )
            iter += 1
        elif  isinstance(m, nn.Linear):
            assert m.weight.shape == list(resnet18_parameters[iter].shape), f"shape error 3 iter:{iter} m.shape:{m.weight.shape} parmeterspe:{list(resnet18_parameters[iter].shape)}"
            m.weight.set_value( resnet18_parameters[iter] )
            iter += 1
            
            m.bias.set_value( resnet18_parameters[iter] )
            iter += 1
    
    net.model_info(True)


    pred = net( paddle.ones(shape=[1, 3, 224, 224]) )
    print("pred: ",pred)