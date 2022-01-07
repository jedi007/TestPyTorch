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
        batch_number = batch_index[-1] + 1  # number of batches

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
        #hyp = self.hyp
        self.mosaic = True
        self.augment = False
        if self.mosaic:
            # load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None
        else:
            # load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # letterbox
            shape = self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scale_up=self.augment)
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

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=0.,
                                            translate=0.,
                                            scale=0.,
                                            shear=0.)

            # Augment colorspace
            # augment_hsv(img, h_gain=hyp["hsv_h"], s_gain=hyp["hsv_s"], v_gain=hyp["hsv_v"])

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0-1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        # if self.augment:
        #     # random left-right flip
        #     lr_flip = True  # 随机水平翻转
        #     if lr_flip and random.random() < 0.5:
        #         img = np.fliplr(img)
        #         if nL:
        #             labels[:, 1] = 1 - labels[:, 1]  # 1 - x_center

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]  # 1 - y_center

        labels_out = torch.zeros((nL, 6))  # nL: number of labels
        if nL:
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

def load_mosaic(self, index):
    """
    将四张图片拼接在一张马赛克图像中
    :param self:
    :param index: 需要获取的图像索引
    :return:
    """
    # loads images in a mosaic

    labels4 = []  # 拼接图像的label信息
    s = self.img_size
    # 随机初始化拼接图像的中心点坐标
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    # 从dataset中随机寻找三张图像进行拼接
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    # 遍历四张图像进行拼接
    for i, index in enumerate(indices):
        # load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            # 创建马赛克图像
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将截取的图像区域填充到马赛克图像的相应位置
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels 获取对应拼接图像的labels信息
        # [class_index, x_center, y_center, w, h]
        x = self.labels[index]
        labels = x.copy()  # 深拷贝，防止修改原数据
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            # 计算标注数据在马赛克图像中的坐标(绝对坐标)
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw   # xmin
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh   # ymin
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw   # xmax
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh   # ymax
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    #cv2.imshow("img4",img4)


    # Augment
    # 随机旋转，缩放，平移以及错切
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=0.,
                                  translate=0.,
                                  scale=0.,
                                  shear=0.,
                                  border=-s // 2)  # border to remove

    #cv2.imshow("img4-2",img4)
    #cv2.waitKey(99999999)
    #exit(0)


    return img4, labels4

def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    """随机旋转，缩放，平移以及错切"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    # 给定的输入图像的尺寸(416/512/640)，等于img4.shape / 2
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    # 生成旋转以及缩放矩阵
    R = np.eye(3)  # 生成对角阵
    a = random.uniform(-degrees, degrees)  # 随机旋转角度
    s = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    # 生成平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    # 生成错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        # 对transform后的bbox进行修正(假设变换后的bbox变成了菱形，此时要修正成矩形)
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # [n, 4]

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        # 计算每个box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets

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
        print("targets: ",targets.shape)
        print("targets: ",targets)


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
