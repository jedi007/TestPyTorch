import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.transforms import functional as F
import cv2
 
font = cv2.FONT_HERSHEY_SIMPLEX
 
root = 'D:/work/Study/Data/COCO2017/coco/images/val2017/'
annFile = 'D:/work/Study/Data/COCO2017/coco/annotations/instances_val2017.json'
 
# 定义 coco collate_fn
def collate_fn_coco(batch):
    return tuple(zip(*batch))
 
if __name__ == '__main__':
    # 创建 coco dataset
    coco_det = datasets.CocoDetection(root,annFile,transform=T.ToTensor())

    print("coco_det: ",coco_det)

    # 创建 Coco sampler
    sampler = torch.utils.data.RandomSampler(coco_det)
    print("sampler: ",sampler)

    batch_sampler = torch.utils.data.BatchSampler(sampler, 8, drop_last=True)
    print("batch_sampler: ",batch_sampler)
    
    # 创建 dataloader
    data_loader = torch.utils.data.DataLoader(
            coco_det, batch_sampler=batch_sampler, num_workers=3,
            collate_fn=collate_fn_coco)
    

    # 可视化
    for imgs,labels in data_loader:
        for i in range(len(imgs)):
            bboxes = []
            ids = []
            img = imgs[i]
            labels_ = labels[i]
            for label in labels_:
                bboxes.append([label['bbox'][0],
                label['bbox'][1],
                label['bbox'][0] + label['bbox'][2],
                label['bbox'][1] + label['bbox'][3]
                ])
                ids.append(label['category_id'])
    
            img = img.permute(1,2,0).numpy()
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            for box ,id_ in zip(bboxes,ids):
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),thickness=2)
                cv2.putText(img, text=str(id_), org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1, 
                    thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))
            cv2.imshow('test',img)
            cv2.waitKey()