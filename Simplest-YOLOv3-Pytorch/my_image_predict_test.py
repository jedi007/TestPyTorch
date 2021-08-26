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

    #test====================================================================================================save
    # img_size = 512
    # input_size = (img_size, img_size)

    # img = torch.ones((1, 3, img_size, img_size), device=device)

    # model.eval()
    # net = torch.jit.trace(model, img)
    # net.save('D:/TestData/my_yolov3_jit.pt')
    # exit(0)
    #test====================================================================================================saveover

    #test====================================================================================================
    img_o = cv2.imread('D:/TestData/cocotest.jpg',cv2.IMREAD_COLOR)
    img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]

    print( "img: ",img.shape )
    
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # scale (0, 255) to (0, 1)
    img = img.unsqueeze(0)  # add batch dimension

    t1 = torch_utils.time_synchronized()

    pred = model(img)

    # print("pred.shape: ",pred.shape)
    # print("pred[0][0]: ",pred[0][0])
    # print("pred[0][99]: ",pred[0][99])


    pred = utils.non_max_suppression(pred, conf_thres=0.2, iou_thres=0.4, multi_label=True)[0]


    pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()

    bboxes = pred[:, :4].detach().cpu().numpy()
    scores = pred[:, 4].detach().cpu().numpy()
    classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1

    img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
    
    img_o = np.array(img_o)
    cv2.imshow('detection', img_o[:,:,[2,1,0]])
    key = cv2.waitKey(99000)


if __name__ == "__main__":
    main()
