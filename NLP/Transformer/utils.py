# hugging face的分词器，github地址：https://github.com/huggingface/tokenizers
from tokenizers import Tokenizer

# 定义一个获取文件行数的方法。
def get_row_count(filepath):
    count = 0
    for _ in open(filepath, encoding='utf-8'):
        count += 1
    return count

# 定义一个获取文件行数的方法。
def get_row_count(filepath):
    count = 0
    for _ in open(filepath, encoding='utf-8'):
        count += 1
    return count


# 加载基础的分词器模型，使用的是基础的bert模型。`uncased`意思是不区分大小写
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
def en_tokenizer(line):
    """
    定义英文分词器，后续也要使用
    :param line: 一句英文句子，例如"I'm learning Deep learning."
    :return: subword分词后的记过，例如：['i', "'", 'm', 'learning', 'deep', 'learning', '.']
    """
    # 使用bert进行分词，并获取tokens。add_special_tokens是指不要在结果中增加‘<bos>’和`<eos>`等特殊字符
    return tokenizer.encode(line, add_special_tokens=False).tokens

from tqdm import tqdm
def yield_en_tokens(en_filepath, row_count):
    """
    每次yield一个分词后的英文句子，之所以yield方式是为了节省内存。
    如果先分好词再构造词典，那么将会有大量文本驻留内存，造成内存溢出。
    """
    file = open(en_filepath, encoding='utf-8')
    print("-------开始构建英文词典-----------")
    for line in tqdm(file, desc="构建英文词典", total=row_count):
        yield en_tokenizer(line)
    file.close()


def zh_tokenizer(line):
    """
    定义中文分词器
    :param line: 中文句子，例如：机器学习
    :return: 分词结果，例如['机','器','学','习']
    """
    return list(line.strip().replace(" ", ""))


def yield_zh_tokens(zh_filepath, row_count):
    file = open(zh_filepath, encoding='utf-8')
    for line in tqdm(file, desc="构建中文词典", total=row_count):
        yield zh_tokenizer(line)
    file.close()