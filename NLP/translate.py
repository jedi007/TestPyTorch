"""
教程:
将法语翻译成英语
"""
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from tqdm import tqdm
import time
import math
from loguru import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS  # 总单词数

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
# 文件都是Unicode编码的，我们需要简单得将字符转化为ASCII编码，全部转化为小写字母，并修剪大部分标点符号。
def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicode2Ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# 读取数据，我们将文件分割成行，进一步分割成句子对。文件是 English->其他语言 ，所以这里加上一个 reverse 参数，可以用来获取 其他语言->英语 的翻译对。
def readLangs(lang1, lang2, reverse=False):
    logger.info("Reading lines...")
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = []
    for l in tqdm(lines[:100]):  # 读取前100行
        l_p = []
        for s in l.split("\t"):
            l_p.append(normalizeString(s))
        pairs.append(l_p)
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines] # 原始语句

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


# 由于样本句子很多，而我们想加速训练。我们会将数据集修剪成相对简短的句子。
# 这里的最大长度是10(包括结束标点)
MAX_LENGTH = 10

# 过滤出 "我是","他是" 这种形式的句子(别忘记之前忽略的撇号,比如 "I'm" 这种形式)
eng_prefixes = (
    "i am", "i m",
    "he is", "he s",
    "she is", "she s",
    "you are", "you re",
    "we are", "we re",
    "they are", "they re"
)


def filterPairs(pairs):
    """的"""
    pair_list = []
    for p in pairs:
        if len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes):
            pair_list.append(p)
    return pair_list


# 完整的数据准备流程如下：
# 读取文本文件，按行分割，再将每行分割成语句对
# 归一化文本，过滤内容和长度
# 根据滤出的句子对创建词语列表

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs0 = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs0))
    # pairs = filterPairs(pairs)
    # 剔除一部分数据
    pairs = []
    for p in pairs0:
        if len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes):
            pairs.append(p)

    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # 此处的hidden_size也是embedding_dim
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_token, hidden):
        # 训练时，hidden输入全0
        # input维度为(seq_len=1,batch_size=1)
        embedded = self.embedding(input_token).view(1, 1, -1)
        # 经过embedded,维度为(seq_len=1,batch_size=1,embedding_dim)
        output = embedded
        # hidden维度为(num_layers*direction_num=1, batch_size=1, hidden_size)
        # output的维度为(seq_length=1, batch_size=1, hidden_size)
        output, hidden = self.gru(output, hidden)  # output与hidden都是[1,1,256]个维度
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        # hidden_size也是embedding_dim
        self.hidden_size = hidden_size
        #  ,
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)  # linear 输出 max_length
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_token, hidden, encoder_outputs):
        # input维度为(seq_len=1,batch_size=1)
        # hidden维度为(num_layers*direction_num=1, batch_size=1, hidden_size)
        # 经过embedded,维度为(seq_len=1,batch_size=1,embedding_dim),其中embedding_dim=hidden_size
        embedded = self.embedding(input_token).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # attn_weights维度为(batch_size, max_length)
        attn_weights = F.softmax(self.attn(torch.cat([embedded[0], hidden[0]], 1)), dim=1)
        # encoder_outputs的维度为(max_length, hidden_size)
        # attn_applied维度为(1, batch_size=1, hidden_size)
        # bmm: input(p,m,n) * mat2(p,n,a) -> output(p,m,a)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # 此时output维度为(batch_size=1, 2*hidden_size)
        output = torch.cat([embedded[0], attn_applied[0]], 1)  # torch.cat()用来拼接
        # 此时output维度为(seq_len=1, batch_size=1, hidden_size)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        # 此时output维度为(seq_len=1, batch_size=1, hidden_size)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        # 此时output维度为(batch_size=1, output_size)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    """训练本次的翻译任务"""
    encoder_hidden = encoder.initHidden()
    # 优化器清零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 输入与输出
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # 预先构造一个全0的encoder
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    # 开始encoder部分的训练
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden  # 这里是把encoder的最终输出作为decoder的输入

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False  # 随机的判断是否使用teacher forcing修正

    if use_teacher_forcing:
        # Teacher forcing: Feed the targer as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teaching forcing: use its own predictions as the next input
        for di in range(target_length):
            # 输入是：开始翻译的标识符，encoder的完整输出，encoder部分的每个输出
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=1000, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]  # 随机挑选一部分

    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, n_iters + 1)):  # 开始迭代训练
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]  # 输入的index
        target_tensor = training_pair[1]  # 输出的index

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    hidden_size = 256
    teacher_forcing_ratio = 0.5
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    trainIters(encoder1, attn_decoder1, 7490, print_every=5000)
