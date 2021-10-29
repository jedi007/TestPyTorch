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

from data.datasets.coco import COCODataset
from data.data_augment import TrainTransform
from data.datasets.mosaicdetection import MosaicDetection
# from data import (
#         YoloBatchSampler,
#         DataLoader,
#         InfiniteSampler,
#         worker_init_reset_seed,
#     )


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

    # # data related init
    # self.train_loader = self.exp.get_data_loader(
    #     batch_size=self.args.batch_size,
    #     is_distributed=self.is_distributed,
    #     no_aug=self.no_aug,
    #     cache_img=self.args.cache,
    # )
    
    dataset = COCODataset(
        data_dir="D:\work\Study\Data\COCO2017",
        json_file="instances_val2017.json", #"instances_train2017.json"  为运行研究方便改动val，真实训练该启用tain
        name="train2017",
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


    # self.prefetcher = DataPrefetcher(self.train_loader)
    # # max_iter means iters per epoch
    # self.max_iter = len(self.train_loader)

    # self.lr_scheduler = self.exp.get_lr_scheduler(
    #     self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
    # )
    # if self.args.occupy:
    #     occupy_mem(self.local_rank)

    # if self.is_distributed:
    #     model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

    # if self.use_model_ema:
    #     self.ema_model = ModelEMA(model, 0.9998)
    #     self.ema_model.updates = self.max_iter * self.start_epoch

    # self.model = model
    # self.model.train()

    # self.evaluator = self.exp.get_evaluator(
    #     batch_size=self.args.batch_size, is_distributed=self.is_distributed
    # )
    # # Tensorboard logger
    # if self.rank == 0:
    #     self.tblogger = SummaryWriter(self.file_name)

    # logger.info("Training start...")
    # logger.info("\n{}".format(model))