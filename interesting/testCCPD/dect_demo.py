import sys
sys.path.append("./YoloV7Core")
sys.path.append("./RCNNCore")

import torch
import random
import cv2
import numpy as np

from YoloV7Core.models.yolo import Model as YOLOModel
from YoloV7Core.utils.general import non_max_suppression, scale_coords
from YoloV7Core.utils.plots import plot_one_box
from YoloV7Core.utils.datasets import letterbox

from RCNNCore.infer import infer_one_img as rcnn_infer_one_img
from RCNNCore.model import Model as RCNNModel
from RCNNCore.tools import words_list


def demo_img(model_yolov7, model_rcnn, img0, device):
    print("demo_img")
    # Padded resize
    img = letterbox(img0, imgsz, stride=32)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model_yolov7(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if int(cls) == 80: # names[80] == 'License plate'
                    label = f'{names[int(cls)]} {conf:.2f}'

                    license_plate = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

                    print("license_plate.shape: ",license_plate.shape)
                    h = license_plate.shape[0]
                    w = license_plate.shape[1]
                    ratio = min(32/h, 130/w)
                    nh = round(h*ratio)
                    nw = round(w*ratio)
                    dh = (32 - nh) / 2
                    dw = (130 - nw) / 2


                    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                    
                    license_plate = cv2.resize(license_plate, (nw, nh), interpolation=cv2.INTER_LINEAR)


                    license_plate = cv2.copyMakeBorder(license_plate, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
                    print("license_plate 2: ", license_plate.shape)


                    # 转化为rcnn 格式
                    license_plate = license_plate[:, :, ::-1].transpose(2, 0, 1)
                    license_plate = np.ascontiguousarray(license_plate)
                    license_plate = torch.from_numpy(license_plate)
                    license_plate = license_plate.to(device, non_blocking=True).float() / 255.0

                    

                    print("license_plate: ", license_plate.shape)
                    # exit(0)

                    text = rcnn_infer_one_img(license_plate, model_rcnn, device)

                    label += " : " + text

                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)

        cv2.imshow("detect show", img0)
        cv2.waitKey() 


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    device = torch.device("cpu")
    imgsz = 640
    
    #init model_yolov7
    ckpt = torch.load("YoloV7Core/best_v7.pt", map_location=device)  # load checkpoint
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    
    model_yolov7 = YOLOModel(ch=3)
    model_yolov7.info(verbose=True)
    model_yolov7.load_state_dict(state_dict, strict=False)  # load
    model_yolov7.to(device)
    model_yolov7.eval()

    # Get names and colors
    names = model_yolov7.module.names if hasattr(model_yolov7, 'module') else model_yolov7.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]



    #init model_rcnn
    model_rcnn = RCNNModel(imgH = 32, number_chanel = 3, number_class = len(words_list))

    model_rcnn.load_state_dict(torch.load("RCNNCore/weights/39-0.034.pth"))
    model_rcnn.eval()
    model_rcnn.to(device)

    
    path = "YoloV7Core/inference/images/0416283524904-91_80-135&514_569&634-582&637_137&612_114&509_559&534-0_0_25_16_31_27_27-119-154.jpg"
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, 'Image Not Found ' + path
    demo_img(model_yolov7, model_rcnn, img, device)

    
    path = "YoloV7Core/inference/images/01-90_265-231&522_405&574-405&571_235&574_231&523_403&522-0_0_3_1_28_29_30_30-134-56.jpg"
    img1 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img1 is not None, 'Image Not Found ' + path
    demo_img(model_yolov7, model_rcnn, img1, device)

    path = "YoloV7Core/inference/images/test.jpeg"
    img2 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img2 is not None, 'Image Not Found ' + path
    demo_img(model_yolov7, model_rcnn, img2, device)
    
    
    path = "YoloV7Core/inference/images/bus.jpg"
    img3 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img3 is not None, 'Image Not Found ' + path
    demo_img(model_yolov7, model_rcnn, img3, device)
