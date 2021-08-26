import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
img = Image.open("D:/TestData/test_rose.jpg")


#transforms.RandomResizedCrop(224) 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小）
#默认scale=(0.08, 1.0)
print("原图大小：",img.size)
data1 = transforms.RandomResizedCrop(224)(img)
print("随机裁剪后的大小:",data1.size)
data2 = transforms.RandomResizedCrop(224)(img)
data3 = transforms.RandomResizedCrop(224)(img)

plt.subplot(2,2,1),plt.imshow(img),plt.title("原图")
plt.subplot(2,2,2),plt.imshow(data1),plt.title("转换后的图1")
plt.subplot(2,2,3),plt.imshow(data2),plt.title("转换后的图2")
plt.subplot(2,2,4),plt.imshow(data3),plt.title("转换后的图3")
plt.show()


#transforms.ToTensor() 将给定图像转为Tensor,其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可
print("111 img: ",img)
print("111 img: ",np.array(img))
img = transforms.ToTensor()(img)
print("after transforms.ToTensor img: \n",img)


#transforms.Normalize(） 使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
# img = Image.open("./demo.jpg")
# img = transforms.ToTensor()(img)
img = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])(img)
print("after transforms.Normalize img: \n",img)
