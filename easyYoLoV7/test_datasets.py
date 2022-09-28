from utils.datasets import create_dataloader
import numpy as np
import torch

from utils.general import *
from utils.draw_box_utils import *

import matplotlib.pyplot as plt
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--img-size', nargs='+', type=int, default=640, help='[train, test] image sizes')
    parser.add_argument('--trainpath', type=str, default='D:/TestData/coco128/train.txt', help='*.data path')
    
    opt = parser.parse_args()
    print("opt: ",opt)


    # class names
    names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush' ]
    names = { i+1:names[i] for i in range(len(names))}
    print("names: ", names)


    hyp = { "mosaic": 1.0,
            "degrees": 0.0,  # image rotation (+/- deg)
            "translate": 0.2,  # image translation (+/- fraction)
            "scale": 0.9,  # image scale (+/- gain)
            "shear": 0.0,  # image shear (+/- deg)
            "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
            "mixup": 0.15,
            "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4, # image HSV-Value augmentation (fraction)
            "paste_in": 0.15,  # image copy paste (probability), use 0 for faster training
            "flipud": 0.0,  # image flip up-down (probability)
            "fliplr": 0.5  # image flip left-right (probability)
          }

    # Trainloader
    dataloader, dataset = create_dataloader(opt.trainpath, opt.img_size, opt.batch_size, 32, opt,
                                            hyp = hyp,
                                            augment=True, world_size=1, workers=opt.workers,
                                            prefix='train: ')
    
    for i, (imgs, targets, paths, _) in enumerate(dataloader):
        print("i: ", i)
        print("targets shape: ",targets.shape)
        #print("targets: ",targets)

        for i in range(opt.batch_size):
            img_o = imgs[i]

            target = targets[targets[:,0]==i]

            bboxes = target[:, 2:].detach().cpu().numpy()*img_o.shape[1]
            bboxes = xywh2xyxy(bboxes)

            scores = torch.ones_like(target[:,1]).cpu().numpy()
            classes = target[:, 1].detach().cpu().numpy().astype(np.int) + 1


            #img_o = img_o.permute(1, 2, 0).numpy()[:,:,[2,1,0]]
            img_o = img_o.permute(1, 2, 0).numpy()
            img_o = draw_box( img_o[:, :, ::-1], bboxes, classes, scores, names)

            img_o = np.array(img_o)
            cv2.imshow('detection', img_o)
            key = cv2.waitKey(0)
            if(key & 0xFF == ord('q')):
                break