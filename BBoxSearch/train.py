from samplest_yolov3_gray import *
from ImagesAndLabelsSet import *
import cv2
from build_utils.utils import *

import torch.optim as optim
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--data', type=str, default='D:/TestData/data/my_data.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='D:/TestData/weights/yolov3spp-voc-512.pt',
                        help='initial weights path')


    opt = parser.parse_args()
    device = torch.device("cuda:0")

    train_path = "D:/TestData/data/my_train_data.txt"


    train_dataset = ImagesAndLabelsSet(train_path, 512, batch_size=opt.batch_size )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=1,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle= False,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)


    model = YOLOv3Model_Gray()
    model.to(device)
    model.train()


# giou: 3.54  # giou loss gain
# cls: 37.4  # cls loss gain
# cls_pw: 1.0  # cls BCELoss positive_weight
# obj: 64.3  # obj loss gain (*=img_size/320 if img_size != 320)
# obj_pw: 1.0  # obj BCELoss positive_weight
# iou_t: 0.20  # iou training threshold
# lr0: 0.001  # initial learning rate (SGD=5E-3 Adam=5E-4)
# lrf: 0.01  # final OneCycleLR learning rate (lr0 * lrf)
# momentum: 0.937  # SGD momentum
# weight_decay: 0.0005  # optimizer weight decay
# fl_gamma: 0.0  # focal loss gamma (efficientDet default is gamma=1.5)
# hsv_h: 0.0138  # image HSV-Hue augmentation (fraction)
# hsv_s: 0.678  # image HSV-Saturation augmentation (fraction)
# hsv_v: 0.36  # image HSV-Value augmentation (fraction)
# degrees: 0.  # image rotation (+/- deg)
# translate: 0.  # image translation (+/- fraction)
# scale: 0.  # image scale (+/- gain)
# shear: 0.  # image shear (+/- deg)
    # optimizer
    parameters_grad = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters_grad, lr=0.001, momentum=0.937,
                          weight_decay=0.005, nesterov=True)

    start_epoch = 0
    epochs = 10
    for epoch in range(start_epoch, epochs):

        for i, (imgs, targets, paths, _, _) in enumerate(train_dataloader):
        
            inputs = torch.zeros( (imgs.shape[0],1,imgs.shape[2],imgs.shape[3]),device=device )
            for img_index in range( imgs.shape[0] ):
                img_o = imgs[img_index]

                img_numpy = img_o.permute(1, 2, 0).numpy()[:,:,[2,1,0]]
                img_gray = cv2.cvtColor(img_numpy,cv2.COLOR_BGR2GRAY)

                inputs[img_index] = torch.tensor(img_gray.reshape(1,1,512,512),device=device)/255.
                

            #print("inputs.shape: ",inputs.shape)
            pred = model( inputs )
            #print( "pred.shape: ",pred.shape )
            #print( "pred: ",pred )

            now_lr = optimizer.param_groups[0]["lr"]

            loss = compute_loss(pred, targets, model)
            
            if i % 20 == 0:
                print("time: {}  epoch: {} step: {}  now_lr: {} loss: {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),epoch, i, now_lr, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        


        #exit(0)