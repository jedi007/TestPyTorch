import torchvision.datasets as datasets
import torchvision
import torch
import numpy as np
import cv2

voc_trainset = datasets.VOCDetection('D:/work/Study/Data',year='2012', image_set='train', download=False)

print('-'*40)
print('VOC2012-trainval')
print(len(voc_trainset))

type(voc_trainset)

o1=next(iter(voc_trainset))
print(o1)

# for i, sample in enumerate(voc_trainset, 1):
#     image, annotation = sample[0], sample[1]['annotation']
#     objects = annotation['object']
#     show_image = np.array(image)
#     print('{} object:{}'.format(i, len(objects)))
#     print(show_image.shape)

def show_object_rect(image: np.ndarray, bndbox):
    pt1 = bndbox[:2]
    pt2 = bndbox[2:]
    image_show = image
    return cv2.rectangle(image_show, pt1, pt2, (0,255,255), 2)


def show_object_name(image: np.ndarray, name: str, p_tl):
    return cv2.putText(image, name, p_tl, 1, 1, (255, 0, 0))
 
for i, sample in enumerate(voc_trainset, 1):
    image, annotation = sample[0], sample[1]['annotation']
    objects = annotation['object']
    show_image = np.array(image)
    print('{} object:{}'.format(i, len(objects)))
    if not isinstance(objects,list):
        object_name = objects['name']
        object_bndbox = objects['bndbox']
        x_min = int(object_bndbox['xmin'])
        y_min = int(object_bndbox['ymin'])
        x_max = int(object_bndbox['xmax'])
        y_max = int(object_bndbox['ymax'])
        show_image = show_object_rect(show_image, (x_min, y_min, x_max, y_max))
        show_image =show_object_name(show_image, object_name, (x_min, y_min))
    else:
        for j in objects:
            object_name = j['name']
            object_bndbox = j['bndbox']
            x_min = int(object_bndbox['xmin'])
            y_min = int(object_bndbox['ymin'])
            x_max = int(object_bndbox['xmax'])
            y_max = int(object_bndbox['ymax'])
            show_image = show_object_rect(show_image, (x_min, y_min, x_max, y_max))
            show_image = show_object_name(show_image, object_name, (x_min, y_min))

    cv2.imshow('image', show_image)
    cv2.waitKey(0)


print(voc_trainset)
print('Down load ok')