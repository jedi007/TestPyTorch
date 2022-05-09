import paddle
import numpy as np

print(paddle.__version__)

# 文件路径
path_to_file = './t8.shakespeare.txt'
test_sentence = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# 文本长度是指文本中的字符个数
print ('Length of text: {} characters'.format(len(test_sentence)))  # Length of text: 5458199 characters

test_sentence_list = test_sentence.lower().split()

word_dict_count = {}
for word in test_sentence_list:
    word_dict_count[word] = word_dict_count.get(word, 0) + 1

# print(word_dict_count.items()) # ... ('disparage', 2), ('aby', 2), ('makes;', 1), ('side?', 3), ...

word_list = []
sorted_word_list = sorted(word_dict_count.items(), key=lambda x: x[1], reverse=True)
print( "sorted_word_list: ", sorted_word_list[0:100] )

for key in sorted_word_list:
    word_list.append(key[0])

word_list = word_list[:2500]
print(len(word_list))  # 使用频率最高的2500个词汇。 由于词表的的长尾，会降低模型训练的速度与精度。



# 设置参数
hidden_size = 1024               # Linear层 参数
embedding_dim = 256              # embedding 维度
batch_size = 256                 # batch size 大小
context_size = 2                 # 上下文长度
vocab_size = len(word_list) + 1  # 词表大小
epochs = 2                       # 迭代轮数


# 将文本被拆成了元组的形式，格式为((‘第一个词’, ‘第二个词’), ‘第三个词’);其中，第三个词就是目标。 当然目标是模拟的数据填入，这里并没有实际意义
trigram = [[[test_sentence_list[i], test_sentence_list[i + 1]], test_sentence_list[i + 2]]
           for i in range(len(test_sentence_list) - 2)]

word_to_idx = {word: i+1 for i, word in enumerate(word_list)}
word_to_idx['<pad>'] = 0
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

# 看一下数据集
print(trigram[:3])

data = trigram[0][0]
label = trigram[0][1]
print("map: ", list(map(lambda word: word_to_idx.get(word, 0), data)) )
data = np.array(list(map(lambda word: word_to_idx.get(word, 0), data)))



class TrainDataset(paddle.io.Dataset):
    def __init__(self, tuple_data):
        self.tuple_data = tuple_data

    def __getitem__(self, idx):
        data = self.tuple_data[idx][0]
        label = self.tuple_data[idx][1]
        data = np.array(list(map(lambda word: word_to_idx.get(word, 0), data)))
        label = np.array(word_to_idx.get(label, 0), dtype=np.int64)
        return data, label
    
    def __len__(self):
        return len(self.tuple_data)
    
train_dataset = TrainDataset(trigram)

# 加载数据
train_loader = paddle.io.DataLoader(train_dataset, return_list=True, shuffle=True, 
                                    batch_size=batch_size, drop_last=True)