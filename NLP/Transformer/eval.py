import torch
import config

from model import *
from utils import *


def translate(model, en_vocab, zh_vocab, src: str):
    """
    :param src: 英文句子，例如 "I like machine learning."
    :return: 翻译后的句子，例如：”我喜欢机器学习“
    """

    # 将与原句子分词后，通过词典转为index，然后增加<bos>和<eos>
    src = torch.tensor([0] + en_vocab(en_tokenizer(src)) + [1]).unsqueeze(0).to(config.device)
    # 首次tgt为<bos>
    tgt = torch.tensor([[0]]).to(config.device)
    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(config.max_length):
        # 进行transformer计算
        out = model(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])
        # print("\npredict: ", predict)
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # print("\ny: ", y)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # print("\ntgt: ", tgt)
        # 如果为<eos>，说明预测结束，跳出循环
        if y == 1:
            break
    # 将预测tokens拼起来
    tgt = ''.join(zh_vocab.lookup_tokens(tgt.squeeze().tolist())).replace("<s>", "").replace("</s>", "")
    return tgt



def run():
    en_vocab = torch.load(config.en_vocab_file, map_location="cpu")
    zh_vocab = torch.load(config.zh_vocab_file, map_location="cpu")


    model = torch.load(config.model_dir / config.model_checkpoint)
    model = model.to(config.device)

    torch.cuda.empty_cache()

    model = model.eval()
    res = translate(model, en_vocab, zh_vocab, "Alright, this project is finished. Let's see how good this is.")
    print("\nres: ", res)

    res = translate(model, en_vocab, zh_vocab, "You like a flower, i love you.")
    print("\nres: ", res)


if __name__ == '__main__':
    run()