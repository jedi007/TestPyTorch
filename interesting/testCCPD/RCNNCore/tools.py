import cv2
import torch
import numpy as np

# [京,沪,津,渝,鲁,冀,鄂,黑,苏,浙,皖,闽,赣,豫,粤,桂,琼,晋,蒙,辽,吉,云,藏,陕,甘,青,宁,湘,川,贵,新,港,澳,台]
# [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# head = ["京", "沪", "津", "渝", "鲁", "冀", "鄂", "黑", "苏", "浙", "皖", "闽", "赣", "豫", "粤", "桂", "琼", "晋", "蒙", "辽",
#  "吉", "云", "藏", "陕", "甘", "青", "宁", "湘", "川", "贵", "新", "港", "澳", "台"]
head = ["皖", "沪", "津", "渝", "冀",
        "晋", "蒙", "辽", "吉", "黑",
        "苏", "浙", "京", "闽", "赣",
        "鲁", "豫", "鄂", "湘", "粤",
        "桂", "琼", "川", "贵", "云",
        "藏", "陕", "甘", "青", "宁",
        "新"]
letter = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

words_list = head + letter + numbers + ["_", "$"]
words_dict = { s:i for i,s in enumerate(words_list)}
print("words_list: ", words_list)
print("words_list len: ", len(words_list))

def encode(str):
    texts = []
    for s in str:
        if s in words_list:
            texts.append(words_dict[s])
        else:
            texts.append(words_dict["_"])
    return texts

def decode(pred):
    length = pred.size(0) #batch_size
    
    char_list = []
    for i in range(length):
        if pred[i] != len(words_list)-1 and pred[i] != len(words_list)-2 and (not (i > 0 and pred[i - 1] == pred[i])):
            char_list.append(words_list[pred[i]])
    print("char_list: ", char_list)
    return ''.join(char_list)


def load_image(imgpath):
    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, 'Image Not Found ' + imgpath

    # resize
    h = img.shape[0]
    w = img.shape[1]
    ratio = min(32/h, 130/w)
    nh = round(h*ratio)
    nw = round(w*ratio)
    dh = (32 - nh) / 2
    dw = (130 - nw) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border


    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img)
    
    return img