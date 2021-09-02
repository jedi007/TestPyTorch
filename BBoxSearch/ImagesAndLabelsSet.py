import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from build_utils.utils import xyxy2xywh, xywh2xyxy
from build_utils import img_utils, torch_utils, utils

import yaml
import argparse
import matplotlib.pyplot as plt
from draw_box_utils import draw_box
import json


img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


class ImagesAndLabelsSet(Dataset):  # for training/testing
    def __init__(self,
                path,   # 指向data/my_train_data.txt路径或data/my_val_data.txt路径
                # 这里设置的是预处理后输出的图片尺寸
                # 当为训练集时，设置的是训练过程中(开启多尺度)的最大尺寸
                # 当为验证集时，设置的是最终使用的网络大小
                img_size=416,
                batch_size=16,
                rank=-1):
        super(ImagesAndLabelsSet, self).__init__()

        try:
            path = str(Path(path))
            # parent = str(Path(path).parent) + os.sep
            if os.path.isfile(path):  # file
                # 读取对应my_train/val_data.txt文件，读取每一行的图片路劲信息
                with open(path, "r") as f:
                    f = f.read().splitlines()
            else:
                raise Exception("%s does not exist" % path)

            # 检查每张图片后缀格式是否在支持的列表中，保存支持的图像路径
            # img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            self.img_files = [x for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(path, e))

        # 如果图片列表中没有图片，则报错
        img_files_number = len(self.img_files)
        assert img_files_number > 0, "No images found in %s." % (path)

        # 将数据划分到一个个batch中
        batch_index = np.floor(np.arange(img_files_number) / batch_size).astype(np.int)
        # 记录数据集划分后的总batch数
        self.batch_number = batch_index[-1] + 1  # number of batches

        self.img_number = img_files_number  # number of images 图像总数目
        self.batch_index = batch_index  # batch index of image 记录哪些图片属于哪个batch
        self.img_size = img_size  # 这里设置的是预处理后输出的图片尺寸

        # Define labels
        # 遍历设置图像对应的label路径
        # (./my_yolo_dataset/train/images/2009_004012.jpg) -> (./my_yolo_dataset/train/labels/2009_004012.txt)
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.img_files]
        
        # Read image shapes (wh)
        # 查看data文件下是否缓存有对应数据集的.shapes文件，里面存储了每张图像的width, height
        shapes_path = path.replace(".txt", ".shapes")  # shapefile path
        try:
            with open(shapes_path, "r") as f:  # read existing shapefile
                shapes_list = [x.split() for x in f.read().splitlines()]
                # 判断现有的shape文件中的行数(图像个数)是否与当前数据集中图像个数相等
                # 如果不相等则认为是不同的数据集，故重新生成shape文件
                assert len(shapes_list) == img_files_number, "shapefile out of aync"
        except Exception as e:
            # print("read {} failed [{}], rebuild {}.".format(sp, e, sp))
            # tqdm库会显示处理的进度
            # 读取每张图片的size信息
            if rank in [-1, 0]:
                image_files = tqdm(self.img_files, desc="Reading image shapes")
            else:
                image_files = self.img_files
            shapes_list = [ Image.open(f).size for f in image_files ]
            # 将所有图片的shape信息保存在.shape文件中
            np.savetxt(shapes_path, shapes_list, fmt="%g")  # overwrite existing (if any)

        # 记录每张图像的原始尺寸
        self.shapes = np.array(shapes_list, dtype=np.float64)

        # cache labels
        self.imgs = [None] * img_files_number  # n为图像总数
        # label: [class, x, y, w, h] 其中的xywh都为相对值
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * img_files_number
        labels_loaded = False
        mission_number, found_number, empty_number, duplicate_number = 0, 0, 0, 0

        np_labels_path = str(Path(self.label_files[0]).parent) + ".npy"

        if os.path.isfile(np_labels_path):
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == img_files_number:
                # 如果载入的缓存标签个数与当前计算的图像数目相同则认为是同一数据集，直接读缓存
                self.labels = x
                labels_loaded = True

        # 处理进度条只在第一个进程中显示
        if rank in [-1, 0]:
            pbar = tqdm(self.label_files)
        else:
            pbar = self.label_files

        
        # 遍历载入标签文件
        for i, file in enumerate(pbar):
            if labels_loaded is True:
                # 如果存在缓存直接从缓存读取
                label = self.labels[i]
            else:
                # 从文件读取标签信息
                try:
                    with open(file, "r") as f:
                        # 读取每一行label，并按空格划分数据
                        label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except Exception as e:
                    print("An error occurred while loading the file {}: {}".format(file, e))
                    mission_number += 1  # file missing
                    print("exit(0)")
                    exit(0)
                    continue

            # 如果标注信息不为空的话
            if label.shape[0]:
                # 标签信息每行必须是五个值[class, x, y, w, h]
                assert label.shape[1] == 5, "> 5 label columns: %s" % file
                assert (label >= 0).all(), "negative labels: %s" % file
                assert (label[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file

                # 检查每一行，看是否有重复信息
                if np.unique(label, axis=0).shape[0] < label.shape[0]:  # duplicate rows
                    duplicate_number += 1

                self.labels[i] = label
                found_number += 1  # file found
            else:
                empty_number += 1  # file empty

            # 处理进度条只在第一个进程中显示
            if rank in [-1, 0]:
                # 更新进度条描述信息
                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                    found_number, mission_number, empty_number, duplicate_number, img_files_number)
        assert found_number > 0, "No labels found in %s." % os.path.dirname(self.label_files[0]) + os.sep

        # 如果标签信息没有被保存成numpy的格式，且训练样本数大于1000则将标签信息保存成numpy的格式
        if not labels_loaded and img_files_number > 1000:
            print("Saving labels to %s for faster future loading" % np_labels_path)
            np.save(np_labels_path, self.labels)  # save for next time

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # load image
        img, (h0, w0), (h, w) = load_image(self, index)

        # letterbox
        shape = self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scale_up=False)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # load labels
        labels = []
        x = self.labels[index]
        if x.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = x.copy()  # label: class, x, y, w, h
            labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        label_number = len(labels)  # number of labels
        if label_number:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0-1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((label_number, 6))  # nL: number of labels
        if label_number:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, index

    def coco_index(self, index):
        """该方法是专门为cocotools统计标签信息准备，不对图像和标签作任何处理"""
        # load image
        # path = self.img_files[index]
        # img = cv2.imread(path)  # BGR
        # import matplotlib.pyplot as plt
        # plt.imshow(img[:, :, ::-1])
        # plt.show()

        # assert img is not None, "Image Not Found " + path
        # o_shapes = img.shape[:2]  # orig hw
        o_shapes = self.shapes[index][::-1]  # wh to hw

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)

        # load labels
        labels = []
        x = self.labels[index]
        if x.size > 0:
            labels = x.copy()  # label: class, x, y, w, h
        return torch.from_numpy(labels), o_shapes
    
    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, index = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, index

def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]  # orig hw
        # img_size 设置的是预处理后输出的图片尺寸
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized

def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img:
    :param new_shape:
    :param color:
    :param auto:
    :param scale_fill:
    :param scale_up:
    :return:
    """

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cfg', type=str, default='D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/cfg/my_yolov3.cfg', help="*.cfg path")
    parser.add_argument('--data', type=str, default='D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/data/my_data.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--weights', type=str, default='D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/weights/yolov3spp-0.pt',
                        help='initial weights path')

    opt = parser.parse_args()

    train_path = "D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/data/my_train_data.txt"


    train_dataset = ImagesAndLabelsSet(train_path, 512, batch_size=4 )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=4,
                                                   num_workers=1,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle= False,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    for i, (imgs, targets, paths, _, _) in enumerate(train_dataloader):
        for i in range(4):
            img_o = imgs[i]

            target = targets[targets[:,0]==i]

            bboxes = target[:, 2:].detach().cpu().numpy()*img_o.shape[1]
            bboxes = xywh2xyxy(bboxes)

            scores = torch.ones_like(target[:,1]).cpu().numpy()
            classes = target[:, 1].detach().cpu().numpy().astype(np.int) + 1

            basepath = "D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp"
            weights = basepath+"/weights/yolov3spp-voc-512.pt"  # 改成自己训练好的权重文件
            json_path = basepath+"/data/pascal_voc_classes.json"  # json标签文件
            assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
            json_file = open(json_path, 'r')
            class_dict = json.load(json_file)
            category_index = {v: k for k, v in class_dict.items()}

            img_o = img_o.permute(1, 2, 0).numpy()[:,:,[2,1,0]]
            img_o = draw_box( img_o[:, :, ::-1], bboxes, classes, scores, category_index)

            img_o = np.array(img_o)
            cv2.imshow('detection', img_o)
            key = cv2.waitKey(3000000)
            if(key & 0xFF == ord('q')):
                break
        break
