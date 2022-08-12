"""进行模型的训练"""
import config
from model import  *
from dataset import get_dataloader
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ImdbModelTransformer() #ImdbModelGRU()
model.to( device )
optimizer = Adam(model.parameters())


def train(epoch):
    train_dataloader = get_dataloader(train=True)
    bar = tqdm(train_dataloader,total=len(train_dataloader))

    for idx,(input,target) in enumerate(bar):
        optimizer.zero_grad()
        output = model(input.to(device))
        loss = F.nll_loss(output,target.to(device))
        loss.backward()
        optimizer.step()
        bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(epoch,idx,loss.item()))


if __name__ == '__main__':
    for i in range(20):
        train(i)

