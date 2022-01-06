from zipfile import ZipFile
from torch.utils.data import Dataset
import torch
import cv2
import numpy
from pathlib import Path
import os


img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

class COCODataSet(Dataset):

    def __init__(self,path,imgsize):
        super(COCODataSet, self).__init__()

        self.imgsize = imgsize
        
        self.img_files = []
        try:
            path = str(Path(path))

            print("path: ",path)
            # parent = str(Path(path).parent) + os.sep
            if os.path.isdir(path):  # file
                # 读取对应my_train/val_data.txt文件，读取每一行的图片路劲信息
                
                for root,dirs,files in os.walk(path): 
                    # for dir in dirs: 
                    #     print(os.path.join(root,dir))
                    for file in files: 
                        filename = os.path.join(root,file)
                        if os.path.splitext(filename)[-1].lower() in img_formats:
                            self.img_files.append(filename)
                        
                        if len(self.img_files) == 1000:
                            break
            else:
                raise Exception("%s does not exist 222" % path)

        except Exception as e:
            raise FileNotFoundError("Error loading data from {} \nerror: {}".format(path, e))

        # 如果图片列表中没有图片，则报错
        img_files_number = len(self.img_files)
        assert img_files_number > 0, "No images found in %s." % (path)        

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        imagename = self.img_files[item]

        image = cv2.imread(imagename)  # BGR
        
        image = cv2.resize(image, (self.imgsize, self.imgsize), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float() / 255
        image = image.permute(2, 0, 1)
        return image