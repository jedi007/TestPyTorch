from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='C:/COCO'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# 初始化标注数据的 COCO api 
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


#加载并显示指定 id 的图片
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = [324158])
# loadImgs() 返回的是只有一个元素的列表, 使用[0]来访问这个元素
# 列表中的这个元素又是字典类型, 关键字有: ["license", "file_name", 
#  "coco_url", "height", "width", "date_captured", "id"]
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# 加载并显示图片,可以使用两种方式: 1) 加载本地图片, 2) 在线加载远程图片
# 1) 使用本地路径, 对应关键字 "file_name"
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))  

# 2) 使用 url, 对应关键字 "coco_url"
I = io.imread(img['coco_url'])        
plt.axis('off')
plt.imshow(I)
plt.show()


# 加载并显示标注信息
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()


#人体关节点检测数据集标注信息====================================================================

# 为 person keypoints 标注信息创建一个 coco 对象
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

# load and display keypoints annotations
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
plt.show()



#语义分析数据集标注信息====================================================================

# 为 caption 标注信息创建一个 coco 对象
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)
# 加载并打印 caption 标注信息
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()