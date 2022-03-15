import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import cv2
from torchvision.transforms.transforms import RandomInvert

transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.5), (0.5))
     ])

trainset = torchvision.datasets.MNIST(root='D:/Study/GitHub/TestPyTorch/data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=False)

testset = torchvision.datasets.MNIST(root='D:/Study/GitHub/TestPyTorch/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False)



# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = dataiter.next()


    net = Net()

    # img = cv2.imread('D:/TestData/index1.png',cv2.IMREAD_GRAYSCALE)

    # print(img.shape)

    # img_tensor = torch.tensor(img)


    # print(img_tensor.shape)

    
    # tensor_img = torch.tensor(img).view(1,1,28,28)

    # outputs = net(tensor_img)

    # print( outputs )
    # exit(0)




    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            #print( inputs.shape )
            #print( type(inputs) )
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.3f %%' % (100 * correct / total))


    images, labels = dataiter.next()

    print(labels[0])
    imshow(images[0])
    outputs = net(images[0].view(1,1,28,28))
    r, predicted = torch.max(outputs.data, 1)
    print("r: ",r)
    print("predicted: ",predicted)

    torch.save(net, "D:/TestData/mnist_fullmodel.pth")

    model = net.eval()
    net = torch.jit.trace(model, torch.rand(1,1,28,28))
    net.save('D:/mnist_save_script_model.pt')
    output=net(torch.ones(1,1,28,28))
    print(output)


    model_test_load = torch.jit.load('D:/mnist_save_script_model.pt')
    print(model_test_load)
