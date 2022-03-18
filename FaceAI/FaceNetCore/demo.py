import os
current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录
print("current_work_dir: ",current_work_dir)
base_dir = "/".join(current_work_dir.split("/")[0:-1])
print("base_dir: ",base_dir)

import sys
sys.path.append(current_work_dir)
sys.path.append(base_dir)

from PIL import Image
import numpy as np
import cv2


from facenet import Facenet

if __name__ == "__main__":
    model = Facenet()
        
    image_path = current_work_dir + "/img/1_001.jpg"
    try:
        image_1 = Image.open(image_path)
        # image_1 = cv2.imread(image_path)
        
        image_1 = cv2.cvtColor(np.asarray(image_1),cv2.COLOR_RGB2BGR)  # PIL.Image转换成OpenCV格式
        image_1 = Image.fromarray(cv2.cvtColor(image_1,cv2.COLOR_BGR2RGB)) # OpenCV转换成PIL.Image格式
    except:
        print('Image Open Error! Try again!')
    
    print("type(image_1): ", type(image_1))
    print("image_1: ", image_1)

    
    features_1 = model.detect_image(image_1)
    print("type(features_1): ", type(features_1))
    print("features_1.shape: ", features_1.shape)
    print("features_1: ", features_1)

    #np.save("myface1.npy",features_1)
    features_1_read = np.load( base_dir + "/myface1.npy")
    print("features_1_read: ", features_1_read)

    dist = model.get_distance(feature_1=features_1, feature_2=features_1_read)
    print("dist: ",dist)
