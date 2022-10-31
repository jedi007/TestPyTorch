#导入相应的包  
from scipy.cluster.vq import kmeans
import numpy as np  
import os
import cv2


def Kmeans(labels_path, k = 3, iters=30, image_type = ".jpg"):
    wh_points = []
    for root,dirs,files in os.walk(labels_path): 
        for file in files: 
            label_path = os.path.join(root,file).replace("\\","/")

            image_path = label_path.replace("/labels/", "/images/").replace(".txt", image_type)

            if not os.path.exists(image_path):
                print("{image_path} didn't find")
                continue

            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert img is not None, 'Image open failed : ' + image_path
            shape = img.shape
            ih = shape[0]
            iw = shape[1]



            with open(label_path, encoding='utf-8', mode='r') as file:
                content = file.readlines()
                ###逐行读取数据
                for line in content:
                    line = line.strip()
                    label = line.split()
                    lh = float(label[4]) * ih
                    lw = float(label[3]) * iw
                    wh_points.append([lw, lh])
        
    centroid = kmeans(wh_points, k, iter=iters)[0]   
    
    return centroid


if __name__ == '__main__':
    r = Kmeans(labels_path="D:/testProject/TestPyTorch/easyYoLoV7/coco128/labels", k = 3)
    
    print("r: ", r)

