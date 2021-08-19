import os
import json
import time

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from build_utils import img_utils, torch_utils, utils
from samplest_yolov3 import YOLOv3Model
from draw_box_utils import draw_box

import cv2


import queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


#获取视频设备/从视频文件中读取视频帧
cap = cv2.VideoCapture(0)

def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    basepath = "D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp"
    weights = basepath+"/weights/yolov3spp-voc-512.pt"  # 改成自己训练好的权重文件
    json_path = basepath+"/data/pascal_voc_classes.json"  # json标签文件
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    print("device: ",device)

    model = YOLOv3Model()
    model.loadPublicPt(weights,device)
    model.to(device)
    model.eval()

    img_size = 512
    input_size = (img_size, img_size)

    img = torch.ones((1, 3, img_size, img_size), device=device)

    net = torch.jit.trace(model, img)
    net.save('D:/TestData/my_yolov3_jit_cuda2.pt')
    # pred=net(img)
    
    # print( "pred.shape: ", pred.shape )
    # print( "pred[0][0]: ", pred[0][0] )
    # print( "pred[0][768]: ", pred[0][768] )
    # print( "pred[0][3840]: ", pred[0][3840] )


if __name__ == "__main__":
    main()
