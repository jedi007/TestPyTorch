from datasets import MyDataset
import torch
from torch.nn import CTCLoss
import torch.optim as optim
from torch.autograd import Variable

from model import Model
from infer import *



def trainBatch(net, train_iter, criterion, optimizer, device):
    data = train_iter.next()
    images, texts = data
    batch_size = images.size(0)
    images = images.to(device, non_blocking=True).float() / 255.0

    text_list = []
    for l in texts:
        text_list += l
    targets = torch.IntTensor(text_list).to(device=device)




    length = [ len(text) for text in texts ]
    length = torch.IntTensor(length).to(device)
    #print("length: ", length)
    

    preds = net(images)  # seqLength x batchSize x alphabet_size
    #print("pred: ", preds.size())
    preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device) # seqLength x batchSize
    #print("preds_size: ", preds_size)
    cost = criterion(preds.log_softmax(2), targets, preds_size, length) / batch_size
    if torch.isnan(cost):
        print(batch_size,texts)
    else:
        net.zero_grad()
        cost.backward()
        optimizer.step()
    
    return cost


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ",device)
    batch_size = 16
    num_workers = 0
    learn_rate = 0.0002
    epochs = 100

    train_dataset = MyDataset("data/test.txt", imgpath="data/test")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle = True,
        num_workers = num_workers,
        collate_fn = MyDataset.collate_fn)
    
    criterion = CTCLoss(reduction='sum',blank=train_dataset.get_words_count()-1).to(device=device)

    model = Model(imgH = 32, number_chanel = 3, number_class = train_dataset.get_words_count())
    model.load_state_dict(torch.load("weights/11-0.20552054047584534.pth"))
    model.train()
    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr = learn_rate)

    for epoch in range(epochs):
        print('epoch {}....'.format(epoch))
        train_iter = iter(train_loader)
        i = 0
        n_batch = len(train_loader)
        mean_loss = 0
        while i < len(train_loader):
            for p in model.parameters():
                p.requires_grad = True
            model.train()
            cost = trainBatch(model, train_iter, criterion, optimizer, device=device)
            mean_loss = (mean_loss * i + cost.item())/(i+1)
            print('epoch: {} iter: {}/{} Train mean_loss: {:.3f}'.format(epoch, i, n_batch, mean_loss))
            i += 1

        torch.save(model.state_dict(), "./weights/{}-{:.3f}.pth".format(epoch, mean_loss))




        

