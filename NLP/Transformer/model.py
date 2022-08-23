import torch.nn as nn
import torch
import math

import config

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model).to(config.device)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    

# 在nn.Transformer中，mask的-inf表示遮掩，而0表示不遮掩。而key_padding_mask的True表示遮掩，False表示不遮掩。
class TranslationModel(nn.Module):

    def __init__(self, d_model, src_vocab, tgt_vocab, dropout=0.1):
        super(TranslationModel, self).__init__()

        # 定义原句子的embedding
        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        # 定义目标句子的embedding
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
        # 定义posintional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=config.max_length)
        # 定义Transformer
        self.transformer = nn.Transformer(d_model, dropout=dropout, batch_first=True)

        # 定义最后的预测层，这里并没有定义Softmax，而是把他放在了模型外。
        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        """
        进行前向传递，输出为Decoder的输出。注意，这里并没有使用self.predictor进行预测，
        因为训练和推理行为不太一样，所以放在了模型外面。
        :param src: 原batch后的句子，例如[[0, 12, 34, .., 1, 2, 2, ...], ...]
        :param tgt: 目标batch后的句子，例如[[0, 74, 56, .., 1, 2, 2, ...], ...]
        :return: Transformer的输出，或者说是TransformerDecoder的输出。
        """

        """
        生成tgt_mask，即阶梯型的mask，例如：
        [[0., -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]]
        tgt.size()[-1]为目标句子的长度。
        """
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(config.device)
        # 掩盖住原句子中<pad>的部分，例如[[False,False,False,..., True,True,...], ...]
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)
        # 掩盖住目标句子中<pad>的部分
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        return tokens == 2