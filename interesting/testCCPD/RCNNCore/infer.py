import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
import random

from model import Model
from tools import decode, load_image

# copy from mydataset
class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.LANCZOS, is_test=True):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.is_test = is_test

    def __call__(self, img):
        w, h = self.size
        w0 = img.size[0]
        h0 = img.size[1]
        if w <= (w0 / h0 * h):
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
        else:
            w_real = int(w0 / h0 * h)
            img = img.resize((w_real, h), self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            tmp = torch.zeros([img.shape[0], h, w])
            start = random.randint(0, w - w_real - 1)
            if self.is_test:
                start = 0
            tmp[:, :, start:start + w_real] = img
            img = tmp
        return img

def infer_one_img(img, model, device):
    image = img.view( 1, *img.shape ).to(device)

    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    txt = decode(preds)
    return txt


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ",device)

    model = Model(imgH = 32, number_chanel = 3, number_class = 72)

    model.load_state_dict(torch.load("22-0.162.pth"))
    model.eval()
    model.to(device)

    # img = load_image("./data/test/3542-湘UBJZHL.jpg")
    # img = img.to(device, non_blocking=True).float() / 255.0
    # text = infer_one_img(img, model, device)
    # print("text: ", text)


    # img = load_image("./data/test/9-甘KCRA5Y.jpg")
    # img = img.to(device, non_blocking=True).float() / 255.0
    # text = infer_one_img(img, model, device)
    # print("text: ", text)


    img = load_image("./data/train/0-浙NJVJLH.jpg")
    img = img.to(device, non_blocking=True).float() / 255.0
    text = infer_one_img(img, model, device)
    print("text: ", text)