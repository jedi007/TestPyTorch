import os

from torchvision.transforms.transforms import RandomInvert
current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录
#current_work_dir = "/".join(current_work_dir.split("/")[0:-1])
print(current_work_dir)

import sys
sys.path.append(current_work_dir)

from models import yolox
import torch
import torch.nn as nn
import time

from data.datasets.coco import COCODataset
from data.data_augment import TrainTransform
from data.datasets.mosaicdetection import MosaicDetection
from data.samplers import InfiniteSampler,YoloBatchSampler
from data.dataloading import DataLoader,worker_init_reset_seed
from data.data_prefetcher import DataPrefetcher

input_size = (416,416)
def preprocess(inputs, targets, tsize):
    scale = tsize[0] / input_size[0]
    if scale != 1:
        inputs = nn.functional.interpolate(
            inputs, size=tsize, mode="bilinear", align_corners=False
        )
        targets[..., 1:] = targets[..., 1:] * scale
    return inputs, targets

if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    model = yolox.YOLOX() #默认相当于yolox-l
    model.to(device)

    
    #=====================================init optimizer============================
    lr = 0.001
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = torch.optim.SGD(
        pg0, lr=lr, momentum=0.9, nesterov=True
    )
    optimizer.add_param_group(
        {"params": pg1, "weight_decay": 1e-5}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})
    #=====================================init optimizer============================ over


    # value of epoch will be set in `resume_train`
    # model = self.resume_train(model) #加载保存的权值
    

    #=====================================init data_loader============================
    dataset = COCODataset(
        data_dir="D:\work\Study\Data\COCO2017",
        json_file="instances_val2017.json", #"instances_train2017.json"  为运行研究方便改动val，真实训练该启用tain
        name="val2017",
        img_size=(416, 416),
        preproc=TrainTransform(max_labels=50),
        cache=False,
    )

    dataset = MosaicDetection(
        dataset,
        mosaic= True,
        img_size=(416, 416),
        preproc=TrainTransform(max_labels=120)
    )

    sampler = InfiniteSampler(len(dataset), seed=0)

    batch_sampler = YoloBatchSampler(
        sampler=sampler,
        batch_size=2,
        drop_last=False,
        mosaic=True,
    )

    data_num_workers = 1
    dataloader_kwargs = {"num_workers": 1, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler

    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    train_loader = DataLoader(dataset, **dataloader_kwargs)
    #=====================================init data_loader============================over

    prefetcher = DataPrefetcher(train_loader)  #据说是增加加载速度
    max_iter = len(train_loader)

    start_epoch = 0
    max_epoch = 300
    no_aug_epochs=15
    from utils import LRScheduler
    lr_scheduler = LRScheduler(
        "yoloxwarmcos",
        lr,
        max_iter,
        max_epoch,
        warmup_epochs=5,
        warmup_lr_start=0,
        no_aug_epochs=no_aug_epochs,
        min_lr_ratio=0.05,
    )
    
    model.train()

    # self.evaluator = self.exp.get_evaluator(
    #     batch_size=self.args.batch_size, is_distributed=self.is_distributed
    # )

    fp16 = False
    data_type = torch.float16 if fp16 else torch.float32
    for epoch in range(start_epoch, max_epoch):
        #self.before_epoch()
        if epoch + 1 == max_epoch - no_aug_epochs:
            train_loader.close_mosaic()
            model.head.use_l1 = True
            eval_interval = 1

        #self.train_in_iter()
            #self.before_iter()
            #pass
        
            #self.train_one_iter()
        iter_start_time = time.time()
        inps, targets = prefetcher.next()
        print("inps:",inps.shape)
        print("targets:",targets.shape)
        inps = inps.to(data_type)
        targets = targets.to(data_type)
        targets.requires_grad = False

        inps, targets = preprocess(inps, targets, (640, 640))
        print("inps:",inps.shape)
        print("targets:",targets.shape)

            #self.after_iter()



        #self.after_epoch()
        exit(0)

    
    #self.after_train()