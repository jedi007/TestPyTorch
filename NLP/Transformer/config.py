from pathlib import Path
import os
import torch

# 工作目录，缓存文件盒模型会放在该目录下
work_dir = Path("./dataset")
# 训练好的模型会放在该目录下
model_dir = Path("./transformer_checkpoints")

# 如果工作目录不存在，则创建一个
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# 如果工作目录不存在，则创建一个
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# 上次运行到的地方，如果是第一次运行，为None，如果中途暂停了，下次运行时，指定目前最新的模型即可。
model_checkpoint = 'model_10000.pt' #None # 'model_10000.pt'

# 英文句子的文件路径
en_filepath = './dataset/train.en'
# 中文句子的文件路径
zh_filepath = './dataset/train.zh'


# 定义句子最大长度，如果句子不够这个长度，则填充，若超出该长度，则裁剪
max_length = 72
# 定义batch_size，由于是训练文本，占用内存较小，可以适当大一些
batch_size = 2
# epochs数量，不用太大，因为句子数量较多
epochs = 1
# 多少步保存一次模型，防止程序崩溃导致模型丢失。
save_after_step = 5000

# 是否使用缓存，由于文件较大，初始化动作较慢，所以将初始化好的文件持久化
use_cache = True

# 定义训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 指定中英文词典缓存文件路径
zh_vocab_file = work_dir / "vocab_zh.pt"
en_vocab_file = work_dir / "vocab_en.pt"