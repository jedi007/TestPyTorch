import datetime
import argparse
import sys

import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from samplest_yolov3 import YOLOv3Model
from build_utils.datasets import *
from build_utils.utils import *
import matplotlib.pyplot as plt
from ImagesAndLabelsSet import *


def train(hyp):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Using {} device training.".format(device.type))

    weightsdir = "weights" + os.sep  # weights dir
    best = weightsdir + "best.pt"

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale

    train_path = "D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/data/my_train_data.txt"
    test_path = "D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/data/my_val_data.txt"

    model = YOLOv3Model().to(device)
    # if weights.endswith(".pt") or weights.endswith(".pth"):
    #     model.loadPublicPt(weights,device)

    # optimizer
    parameters_grad = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters_grad, lr=hyp["lr0"], momentum=hyp["momentum"],
                          weight_decay=hyp["weight_decay"], nesterov=True)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)

    start_epoch = 0
    best_map = 0.0

    train_dataset = ImagesAndLabelsSet(train_path, 512, batch_size = opt.batch_size )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size = opt.batch_size,
                                                   num_workers=1,
                                                   shuffle= False,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    # Model parameters
    model.class_number = 20  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)

    for epoch in range(start_epoch, epochs):
        mloss, lr = train_one_epoch(model, optimizer, train_dataloader,
                                    device, epoch,
                                    img_size=imgsz_train,print_freq=1,)

        
        # update scheduler
        scheduler.step()
    
        if True:
            if opt.savebest is False:
                save_data = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_map': best_map}
                torch.save(save_data, "D:/TestData/yolov3spp-{}-{}.pt".format(epoch,mloss[3]))



    

def train_one_epoch(model, optimizer, data_loader, device, epoch,img_size,print_freq=10): # print_freq 每训练多少个step打印一次信息
    model.train()

    mloss = torch.zeros(4).to(device)  # mean losses

    for i, (imgs, targets, paths, _, _) in enumerate( data_loader ):
        # ni 统计从epoch0开始的所有batch数
        #ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        pred = model(imgs)

        # loss
        loss_dict = compute_loss(pred, targets, model)
        
        losses = sum(loss for loss in loss_dict.values())

        loss_items = torch.cat((loss_dict["box_loss"],
                                loss_dict["obj_loss"],
                                loss_dict["class_loss"],
                                losses)).detach()
        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

        if not torch.isfinite(losses):
            print("in imgs[0]: ",imgs[0])
            print("pred: ",pred[0][0][0])

            print('WARNING: non-finite loss, ending training ', loss_dict)
            print("training image path: {}".format(",".join(paths)))
            sys.exit(1)

        #losses *= 1. / accumulate  # scale loss

        losses.backward()
        optimizer.step()

        optimizer.zero_grad()

        now_lr = optimizer.param_groups[0]["lr"]

        print("epoch: {} step: {}  now_lr: {} loss_items: {}".format(epoch, i, now_lr, loss_items))

    return mloss, now_lr




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)

    parser.add_argument('--cfg', type=str, default='D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/cfg/my_yolov3.cfg', help="*.cfg path")
    parser.add_argument('--data', type=str, default='D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/data/my_data.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--savebest', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp/weights/yolov3spp-0.pt',
                        help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Freeze non-output layers')
    opt = parser.parse_args()

    print("opt.rect: ",opt.rect)

    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)
    print(opt)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    train(hyp)