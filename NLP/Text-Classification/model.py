"""构建模型"""
import torch.nn as nn
import config
import torch.nn.functional as F
import torch
import numpy as np

class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel,self).__init__()

        embedding_dim = 200

        self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=embedding_dim,padding_idx=config.ws.PAD)
        self.fc = nn.Linear(config.max_len*embedding_dim,2)
    
    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """

        #print("input shape: ", input.shape) # torch.Size([512, 50])

        input_embeded = self.embedding(input) #input embeded :[batch_size,max_len,200]

        #print("1 input_embeded shape: ", input_embeded.shape) # torch.Size([512, 50, 200])

        #变形
        input_embeded_viewed = input_embeded.view(input_embeded.size(0),-1)

        #print("2 input_embeded_viewed shape: ", input_embeded_viewed.shape) # torch.Size([512, 10000])

        #全连接
        out = self.fc(input_embeded_viewed)
        #print("out shape: ", out.shape) # torch.Size([512, 2])

        #print("out : ", out)
        #print("out softmax: ", F.log_softmax(out,dim=-1))


        return F.log_softmax(out,dim=-1)


class ImdbModelRNN(nn.Module):
    def __init__(self):
        super(ImdbModelRNN,self).__init__()

        embedding_dim = 200
        hidden_size = 256
        bidirectional = True
        scale = 2 if bidirectional else 1

        self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=embedding_dim,padding_idx=config.ws.PAD)

        self.rnn = nn.RNN(batch_first=True, input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True, nonlinearity="tanh", bidirectional=bidirectional)  
        #torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity="tanh", bias=True)

        self.rnn_fc1 = nn.Linear(config.max_len*hidden_size*scale,hidden_size)
        self.rnn_fc2 = nn.Linear(hidden_size,2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """

        input_embeded = self.embedding(input)

        rnn_res, h = self.rnn(input_embeded)
        # print("rnn_res: ", rnn_res.shape) #torch.Size([512, 50, 512])

        input_embeded_viewed = rnn_res.contiguous().view(rnn_res.size(0),-1)
        #print("rnn input_embeded_viewed : ", input_embeded_viewed.shape) #torch.Size([512, 25600])

        out1 = self.rnn_fc1(input_embeded_viewed)
        #print("out1 shape: ", out1.shape) #torch.Size([512, 512])

        out2 = self.rnn_fc2(out1)
        #print("out2 shape: ", out2.shape) #torch.Size([512, 2])

        return F.log_softmax(out2,dim=-1)


class ImdbModelLSTM(nn.Module):
    def __init__(self):
        super(ImdbModelLSTM,self).__init__()

        embedding_dim = 200
        hidden_size = 256
        bidirectional = True
        scale = 2 if bidirectional else 1

        self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=embedding_dim,padding_idx=config.ws.PAD)

        self.lstm = nn.LSTM(batch_first=True, input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True,  bidirectional=bidirectional) 

        self.rnn_fc1 = nn.Linear(config.max_len*hidden_size*scale,hidden_size)
        self.rnn_fc2 = nn.Linear(hidden_size,2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """

        input_embeded = self.embedding(input)

        rnn_res, h = self.lstm(input_embeded)
        # print("rnn_res: ", rnn_res.shape) #torch.Size([512, 50, 512])

        input_embeded_viewed = rnn_res.contiguous().view(rnn_res.size(0),-1)
        #print("rnn input_embeded_viewed : ", input_embeded_viewed.shape) #torch.Size([512, 25600])

        out1 = self.rnn_fc1(input_embeded_viewed)
        #print("out1 shape: ", out1.shape) #torch.Size([512, 512])

        out2 = self.rnn_fc2(out1)
        #print("out2 shape: ", out2.shape) #torch.Size([512, 2])

        return F.log_softmax(out2,dim=-1)



class ImdbModelGRU(nn.Module):
    def __init__(self):
        super(ImdbModelGRU,self).__init__()

        embedding_dim = 200
        hidden_size = 256
        bidirectional = True
        scale = 2 if bidirectional else 1

        self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=embedding_dim,padding_idx=config.ws.PAD)

        self.gru = nn.GRU(batch_first=True, input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True,  bidirectional=bidirectional) 

        self.rnn_fc1 = nn.Linear(config.max_len*hidden_size*scale,hidden_size)
        self.rnn_fc2 = nn.Linear(hidden_size,2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """

        input_embeded = self.embedding(input)

        rnn_res, h = self.gru(input_embeded)
        # print("rnn_res: ", rnn_res.shape) #torch.Size([512, 50, 512])

        input_embeded_viewed = rnn_res.contiguous().view(rnn_res.size(0),-1)
        #print("rnn input_embeded_viewed : ", input_embeded_viewed.shape) #torch.Size([512, 25600])

        out1 = self.rnn_fc1(input_embeded_viewed)
        #print("out1 shape: ", out1.shape) #torch.Size([512, 512])

        out2 = self.rnn_fc2(out1)
        #print("out2 shape: ", out2.shape) #torch.Size([512, 2])

        return F.log_softmax(out2,dim=-1)



class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(x.device)
        out = self.dropout(out)
        return out

class ImdbModelTransformer(nn.Module):
    def __init__(self):
        super(ImdbModelTransformer,self).__init__()

        embedding_dim = 200

        self.embedding = nn.Embedding(num_embeddings=len(config.ws),embedding_dim=embedding_dim,padding_idx=config.ws.PAD)

        self.postion_embedding = Positional_Encoding(embedding_dim, config.max_len, dropout=0.5)
        
        #Transformer结构有两种：Encoder和Decoder，在文本分类中只使用到了Encoder，Decoder是生成式模型，主要用于自然语言生成的。
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)
        self.trans_encode = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # Examples::
        # >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # >>> src = torch.rand(10, 32, 512)
        # >>> out = transformer_encoder(src)

        self.rnn_fc1 = nn.Linear(config.max_len*embedding_dim,128)
        self.rnn_fc2 = nn.Linear(128,2)

    def forward(self, input):
        """
        :param input:[batch_size,max_len]
        :return:
        """

        input_embeded = self.embedding(input)
        input_postioned = self.postion_embedding(input_embeded)

        trans_res = self.trans_encode(input_postioned)
        # print("trans_res: ", trans_res.shape)  #torch.Size([2, 50, 200])  [batch_size,max_len,embedding_dim]

        input_embeded_viewed = trans_res.contiguous().view(trans_res.size(0),-1)
        # print("rnn input_embeded_viewed : ", input_embeded_viewed.shape)  #torch.Size([2, 10000])

        out1 = self.rnn_fc1(input_embeded_viewed)
        # print("out1 shape: ", out1.shape) #torch.Size([2, 128])

        out2 = self.rnn_fc2(out1)
        # print("out2 shape: ", out2.shape) #torch.Size([2, 2])

        return F.log_softmax(out2,dim=-1)