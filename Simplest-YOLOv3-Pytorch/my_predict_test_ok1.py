import os
import json
import time

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from build_utils import img_utils, torch_utils, utils
from samplest_yolov3_ok1 import YOLOv3Model
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLOv3Model()
    model.loadPublicPt(weights,device)
    
    
    model.to(device)

    model.eval()


    while cap.isOpened():
        #从摄像头读视频帧
        ret, frame = cap.read()

        if ret == True:
            img_o = frame

            with torch.no_grad():
                img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(device).float()
                img /= 255.0  # scale (0, 255) to (0, 1)
                img = img.unsqueeze(0)  # add batch dimension

                t1 = torch_utils.time_synchronized()

                pred = model(img)
                t2 = torch_utils.time_synchronized()
                print(t2 - t1)

                pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
                t3 = time.time()
                print(t3 - t2)

                if pred is None:
                    print("No target detected.")
                    exit(0)

                # process detections
                pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
                print(pred.shape)

                bboxes = pred[:, :4].detach().cpu().numpy()
                scores = pred[:, 4].detach().cpu().numpy()
                classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1

                img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
                
                img_o = np.array(img_o)
                cv2.imshow('detection', img_o[:,:,[2,1,0]])
                
                #等待键盘事件，如果为q，退出
                key = cv2.waitKey(1)
                if(key & 0xFF == ord('q')):
                  cv2.destroyAllWindows()
                  break
    #释放VideoCapture
    cap.release()   
    
if __name__ == "__main__":
    main()
