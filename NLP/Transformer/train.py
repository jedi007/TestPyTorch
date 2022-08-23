import config
import torch
from tqdm import tqdm
from LOSS import *

from model import *
from utils import *
from datasets import *



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
    # 英文句子数量
    en_row_count = get_row_count(config.en_filepath)
    # 中文句子数量
    zh_row_count = get_row_count(config.zh_filepath)
    assert en_row_count == zh_row_count, "英文和中文文件行数不一致！"
    # 句子数量，主要用于后面显示进度。
    row_count = en_row_count

    en_vocab = torch.load(config.en_vocab_file, map_location="cpu")
    zh_vocab = torch.load(config.zh_vocab_file, map_location="cpu")
    # 打印看一下效果
    print("中文词典大小:", len(zh_vocab))
    print(dict((i, zh_vocab.lookup_token(i)) for i in range(10)))
    print("英文词典大小:", len(en_vocab))
    print(dict((i, en_vocab.lookup_token(i)) for i in range(10)))


    dataset = TranslationDataset(config.en_filepath, en_vocab, config.zh_filepath, zh_vocab)
    print(dataset.__getitem__(0))

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)


    if config.model_checkpoint and os.path.exists(config.model_checkpoint):
        model = torch.load(config.model_dir / config.model_checkpoint)
    else:
        model = TranslationModel(256, en_vocab, zh_vocab)
    model = model.to(config.device)


    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    criteria = TranslationLoss()

    torch.cuda.empty_cache()

    if config.model_checkpoint:
        step = int(config.model_checkpoint.replace("model_", "").replace(".pt", ""))

    step = 0

    model.train()
    for epoch in range(config.epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for index, data in enumerate(train_loader):
            # 生成数据
            src, tgt, tgt_y, n_tokens = data
            src, tgt, tgt_y = src.to(config.device), tgt.to(config.device), tgt_y.to(config.device)

            # 清空梯度
            optimizer.zero_grad()
            # 进行transformer的计算
            out = model(src, tgt)
            # 将结果送给最后的线性层进行预测
            out = model.predictor(out)

            """
            计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                    我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
                    (batch_size*词数, 词典大小)。
                    而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                    除以n_tokens。
            """
            loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            loop.set_description("Epoch {}/{}".format(epoch, config.epochs))
            loop.set_postfix(loss=loss.item())
            loop.update(1)

            step += 1

            del src
            del tgt
            del tgt_y

            if step != 0 and step % config.save_after_step == 0:
                torch.save(model, config.model_dir / f"model_{step}.pt")
            
            if step > 0:
                break
    

    model = model.eval()
    res = translate(model, en_vocab, zh_vocab, "Alright, this project is finished. Let's see how good this is.")
    print("\nres: ", res)

if __name__ == '__main__':
    run()