from torch.utils.data import Dataset
from PIL import Image,ImageEnhance,ImageOps
import numpy as np
import torchvision.transforms as transforms
import cv2
import torch

def load_image(imgpath):
    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, 'Image Not Found ' + imgpath

    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.imshow("test", img)
    # cv2.waitKey(0)

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img)
    
    return img

class MyDataset(Dataset):
    # [京,沪,津,渝,鲁,冀,鄂,黑,苏,浙,皖,闽,赣,豫,粤,桂,琼,晋,蒙,辽,吉,云,藏,陕,甘,青,宁,湘,川,贵,新,港,澳,台]
    # [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    head = ["京", "沪", "津", "渝", "鲁", "冀", "鄂", "黑", "苏", "浙", "皖", "闽", "赣", "豫", "粤", "桂", "琼", "晋", "蒙", "辽", "吉", "云", "藏", "陕", "甘", "青", "宁", "湘", "川", "贵", "新", "港", "澳", "台"]
    letter = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def __init__(self, info_filename, imgpath):
        super(Dataset, self).__init__()
        self.info_filename = info_filename
        self.imgpath = imgpath
        self.words_list = self.head + self.letter + self.numbers + ["_", "$"]
        self.words_dict = { s:i for i,s in enumerate(self.words_list)}
        print("self.words_list: ", self.words_list)
        print("self.words_list len: ", len(self.words_list))
        
        self.files = list()
        self.labels = list()

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

        label = self.encode(label)

        return img,label

    def __len__(self):
        return len(self.labels)

    def get_words_count(self):
        return len(self.words_list)
    
    def encode(self, str):
        texts = []
        for s in str:
            if s in self.words_list:
                texts.append(self.words_dict[s])
            else:
                texts.append(self.words_dict["_"])
        return texts
    
    def decode(self, pred):
        length = pred.size(0) #batch_size
        
        char_list = []
        for i in range(length):
            if pred[i] != len(self.words_list)-1 and pred[i] != len(self.words_list)-2 and (not (i > 0 and pred[i - 1] == pred[i])):
                char_list.append(self.words_list[pred[i]])
        print("char_list: ", char_list)
        return ''.join(char_list)
    
    @staticmethod
    def collate_fn(batch):
        imgs, label = zip(*batch)  # transposed
        return torch.stack(imgs, 0), label

if __name__ == '__main__':
    dataset = MyDataset("data/test.txt", imgpath="data/test")
    img, label = dataset.__getitem__(0)
    print("label: ", label)
    # img.show()