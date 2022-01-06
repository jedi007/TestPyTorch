from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
 
path = r'C:/COCO/annotations/instances_train2017.json'
coco = COCO(path)#1.读文件
 
def showImage(imgIds):
    IDs = coco.getImgIds(imgIds)#2.获取ID
    imgs = coco.loadImgs(IDs)[0]#3.根据ID读图，返回list
    print(imgs)
    imgFile = r'C:/COCO/images/train2017/'
    imgURL = imgFile + imgs['file_name']
    I = io.imread(imgURL)#4.读图
    plt.imshow(I)
    plt.show()
    return imgs, I
 
'用类别来查询图片'
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
print('imgIds:', imgIds[3])
imgs, I = showImage(imgIds[3])
 
plt.imshow(I)
annIds = coco.getAnnIds(imgs['id'])
imgAnns = coco.loadAnns(ids=annIds)
coco.showAnns(imgAnns)
plt.show()