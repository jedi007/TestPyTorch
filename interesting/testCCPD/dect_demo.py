from models.yolo import Model as YOLOModel
import torch
import random
import cv2
import numpy as np

from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

from infer import *


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = dw % stride, dh % stride  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def demo_img(model_yolov7, model_rcnn, img0):
    print("demo_img")
    datatool = MyDataset("data/test.txt", imgpath="data/test")

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

                    license_plate = cv2.resize(license_plate, (108, 32), interpolation=cv2.INTER_LINEAR)
                    # top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                    # left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                    license_plate = cv2.copyMakeBorder(license_plate, 0, 0, 11, 11, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
                    print("license_plate 2: ", license_plate.shape)


                    # 转化为rcnn 格式
                    license_plate = license_plate[:, :, ::-1].transpose(2, 0, 1)
                    license_plate = np.ascontiguousarray(license_plate)
                    license_plate = torch.from_numpy(license_plate)
                    license_plate = license_plate.to(device, non_blocking=True).float() / 255.0

                    

                    print("license_plate: ", license_plate.shape)
                    # exit(0)

                    text = infer_one_img(license_plate, model_rcnn, datatool, device)

                    label += " : " + text

                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)

        cv2.imshow("detect show", img0)
        cv2.waitKey() 


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    imgsz = 640
    
    #init model_yolov7
    ckpt = torch.load("best_v7.pt", map_location=device)  # load checkpoint
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
    model_rcnn = Model(imgH = 32, number_chanel = 3, number_class = 72)

    model_rcnn.load_state_dict(torch.load("weights/22-0.162.pth"))
    model_rcnn.eval()
    model_rcnn.to(device)

    
    path = "inference/images/test.jpeg"
    # path = "inference/images/01-90_265-231&522_405&574-405&571_235&574_231&523_403&522-0_0_3_1_28_29_30_30-134-56.jpg"
    # path = "inference/images/bus.jpg"
    img0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img0 is not None, 'Image Not Found ' + path


    demo_img(model_yolov7, model_rcnn, img0)
