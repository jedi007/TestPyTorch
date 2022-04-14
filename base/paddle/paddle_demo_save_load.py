import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D,MaxPool2D,Linear
from paddle.static import InputSpec
from paddle.vision.transforms import ToTensor

print(paddle.__version__)

train_dataset = MNIST(mode='train', transform=ToTensor())
test_dataset = MNIST(mode='test', transform=ToTensor())

class MyModel(Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = Linear(in_features=16*5*5, out_features=120)
        self.linear2 = Linear(in_features=120, out_features=84)
        self.linear3 = Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


inputs = InputSpec([None, 784], 'float32', 'x')
labels = InputSpec([None, 10], 'float32', 'x')
model = paddle.Model(MyModel(), inputs, labels)

optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

model.load("./mnist_checkpoint0/final") # 加载参数恢复训练

model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )


strategy = 1

if strategy == 1:
    # 方法一：训练过程中实时保存每个epoch的模型参数
    # 每个epoch生成两种文件 0.pdparams,0.pdopt，分别存储了模型参数和优化器参数，
    model.fit(train_dataset,
            test_dataset,
            epochs=3,
            batch_size=64,
            save_dir='mnist_checkpoint',
            verbose=1
            )
elif strategy == 2:
    # 方法可以保存模型结构、网络参数和优化器参数，参数training=true的使用场景是在训练过程中，
    # 此时会保存网络参数和优化器参数。每个epoch生成两种文件 0.pdparams,0.pdopt，分别存储了模型参数和优化器参数，
    # 但是只会在整个模型训练完成后才会生成包含所有epoch参数的文件，path的格式为’dirname/file_prefix’ 
    # 或 ‘file_prefix’，其中dirname指定路径名称，file_prefix 指定参数文件的名称。当training=false的时候，代表已经训练结束，此时存储的是预测模型结构和网络参数。

    # 方法二：model.save()保存模型和优化器参数信息
    model.save('mnist_checkpoint/test')