import os
current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录
print("current_work_dir: ",current_work_dir)
base_dir = "/".join(current_work_dir.split("/")[0:-1])
print("base_dir: ",base_dir)

import sys
sys.path.append(current_work_dir)
sys.path.append(base_dir)


import torch
from tools import *


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = current_work_dir + '/weights/yolov5s-face.pt'
    #image_path = current_work_dir + '/data/images/test.jpg'
    image_path = current_work_dir + '/data/images/cocotest.jpg'

    print("weights_path: ",weights_path)

    model = torch.load(weights_path, map_location=device)['model']
    
    
    #print(model)
    #检测图片
    detect_one_img(model, image_path, device)



    #获取视频设备/从视频文件中读取视频帧
    cap = cv2.VideoCapture(0)

    #检测视频
    detect_VideoCapture(cap, model, device)
    
    #释放VideoCapture
    cap.release()   