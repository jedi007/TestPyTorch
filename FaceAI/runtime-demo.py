import os
current_work_dir0 = os.path.dirname(__file__)  # 当前文件所在的目录
print("current_work_dir0: ",current_work_dir0)
base_dir0 = "/".join(current_work_dir0.split("/")[0:-1])
print("base_dir0: ",base_dir0)

import sys
sys.path.append(current_work_dir0)
sys.path.append(base_dir0)


from PIL import Image
import numpy as np
import cv2
import torch


from Yolo5FaceCore.tools import *
from FaceNetCore.facenet import Facenet


def demo_img(model_yolo5face, model_facenet, features_myface):
    print("demo_img")

    #image_path = current_work_dir + '/data/images/test.jpg'
    image_path = current_work_dir0 + '/Yolo5FaceCore/data/images/cocotest.jpg'
    
    #检测图片
    orgimg = cv2.imread(image_path)  # BGR

    _, dets = detect_one_frame(model_yolo5face, copy.deepcopy(orgimg), device)
    
    h,w,c = orgimg.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    for i, det in enumerate(dets):  # detections per image
        for j in range(det.size()[0]):
            xyxy = det[j, :4].view(-1).tolist()
            
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            cv2.rectangle(orgimg, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

            face = orgimg[y1:y2, x1:x2]

            face = Image.fromarray(cv2.cvtColor(face,cv2.COLOR_BGR2RGB)) # OpenCV转换成PIL.Image格式
            features = model_facenet.detect_image(face)

            dist = model_facenet.get_distance(feature_1=features, feature_2=features_myface)
            print("dist: ",dist)


            if dist > 0.8:
                cv2.putText(orgimg, "Unknown", (x1, y1 - 2), 0, tl / 3, [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)
            else:
                cv2.putText(orgimg, "Master", (x1, y1 - 2), 0, tl / 3, [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    cv2.imshow("orgimg", orgimg )
    cv2.waitKey(0)


def demo_VideoCapture(model_yolo5face, model_facenet, features_myface):
    #获取视频设备/从视频文件中读取视频帧
    cap = cv2.VideoCapture(0)

    #检测视频
    while cap.isOpened():
        #从摄像头读视频帧
        ret, frame = cap.read()

        if ret == True:
            
            _, dets = detect_one_frame(model_yolo5face, copy.deepcopy(frame), device)
            
            h,w,c = frame.shape
            tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
            tf = max(tl - 1, 1)  # font thickness
            for i, det in enumerate(dets):  # detections per image
                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])

                    cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

                    face = frame[y1:y2, x1:x2]

                    face = Image.fromarray(cv2.cvtColor(face,cv2.COLOR_BGR2RGB)) # OpenCV转换成PIL.Image格式
                    features = model_facenet.detect_image(face)

                    dist = model_facenet.get_distance(feature_1=features, feature_2=features_myface)
                    print("dist: ",dist)


                    if dist > 1.0:
                        cv2.putText(frame, "Unknown", (x1, y1 - 2), 0, tl / 3, [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "Master", (x1, y1 - 2), 0, tl / 3, [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)

            cv2.imshow("VideoCapture", frame)
            
            #等待键盘事件，如果为q，退出
            key = cv2.waitKey(1)
            if(key & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break
    
    #释放VideoCapture
    cap.release()   

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_facenet = Facenet()
    features_myface = np.load( current_work_dir0 + "/myface1.npy")


    weights_path = current_work_dir0 + '/Yolo5FaceCore/weights/yolov5s-face.pt'
    model_yolo5face = torch.load(weights_path, map_location=device)['model']
    
    


    # demo_img(model_yolo5face, model_facenet, features_myface)

    demo_VideoCapture(model_yolo5face, model_facenet, features_myface)

