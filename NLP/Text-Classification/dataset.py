"""
准备数据
"""
from torch.utils.data import DataLoader,Dataset
import torch
import os
import utils
import config



class ImdbDataset(Dataset):
    def __init__(self,train=True):
        # super(ImdbDataset,self).__init__()
        # data_path = r"F:\AI\阶段9-人工智能NLP项目\第四天\代码\data\aclImdb"
        # data_path += r"\train" if train else r"\test"
        # self.total_path = []  #保存所有的文件路径
        # for temp_path in [r"\pos",r"\neg"]:
        #     cur_path = data_path + temp_path
        #     self.total_path += [os.path.join(cur_path,i) for i in os.listdir(cur_path) if i.endswith(".txt")]
        
        self.reviews = ["ha ha he he hei hei ya", "good gooder goodest"]
        self.labels = [0,1]


    def __getitem__(self, idx):
        # file = self.total_path[idx]
        # #with open(file, 'r', encoding='utf-8') as f:
        # review = utils.tokenlize(open(file, 'r', encoding='utf-8').read()) #评论
        # label = int(file.split("_")[-1].split(".")[0])
        # label = 0 if label <5 else 1
        # return review,label
        return [i for i in self.reviews[idx].split(" ") if len(i)>0], self.labels[idx]

    def __len__(self):
        #return len(self.total_path)
        return len(self.reviews)


def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews,labels = zip(*batch)
    reviews = torch.LongTensor([config.ws.transform(i,max_len=config.max_len) for i in reviews])
    labels = torch.LongTensor(labels)

    return reviews,labels


def get_dataloader(train=True):
    dataset = ImdbDataset(train)
    batch_size = config.train_batch_size if train else config.test_batch_size
    return DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)



if __name__ == '__main__':
    for idx,(review,label) in enumerate(get_dataloader(train=True)):
        print(idx)
        print(review)
        print(label)
        break