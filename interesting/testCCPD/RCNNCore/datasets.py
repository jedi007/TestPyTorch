from torch.utils.data import Dataset
from PIL import Image,ImageEnhance,ImageOps
import numpy as np
import torchvision.transforms as transforms
import cv2
import torch

from tools import *

class MyDataset(Dataset):
    def __init__(self, info_filename="", imgpath=""):
        super(Dataset, self).__init__()
        self.info_filename = info_filename
        self.imgpath = imgpath
        
        self.files = list()
        self.labels = list()

        if len(info_filename) == 0:
            return
        with open(info_filename, 'r', encoding='utf-8') as f:
            content = f.readlines()
            for line in content:
                fname, label = line.split('\t')
                self.files.append(fname.strip())
                self.labels.append(label.strip())

    def name(self):
        return 'MyDataset'

    def __getitem__(self, index):
        img = load_image(self.imgpath + "/" + self.files[index])
        label = self.labels[index]

        label = encode(label)

        return img,label

    def __len__(self):
        return len(self.labels)
    
    @staticmethod
    def collate_fn(batch):
        imgs, label = zip(*batch)  # transposed
        return torch.stack(imgs, 0), label

if __name__ == '__main__':
    dataset = MyDataset("data/test.txt", imgpath="data/test")
    img, label = dataset.__getitem__(0)
    print("label: ", label)
    # img.show()