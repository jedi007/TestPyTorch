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
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm(64)
        self.relu1 = ReLU()
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm(64)
        self.relu2 = ReLU()
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm(128)
        self.relu3 = ReLU()
        self.maxpool = nn.Pool2D(pool_padding=1,pool_size=3,pool_type="max",pool_stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
  
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0]*m.weight.shape[1]*m.weight.shape[2]
                v = np.random.normal(loc=0.,scale=np.sqrt(2./n),size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

    def _make_layer(self, block, planes, blocks, stride=1):
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
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def resnet50():
    return ResNet(Bottleneck, [3,4,6,3])

if __name__ == "__main__":
    model = paddle.Model(resnet50())
    model.summary((-1, 3, 224, 224))