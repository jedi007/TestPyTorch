# Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
# 使用pip就可以安装。

from tqdm import tqdm

for i in tqdm(range(1000)):  
     #do something
     pass  

for char in tqdm(["a", "b", "c", "d"]):
    #do something
    pass



from tqdm import trange
for i in trange(100):
    #do something
    pass

#在for循环外部初始化tqdm,可以打印其他信息
pbar = tqdm(["a", "b", "c", "d"])
for char in pbar:
    pbar.set_description("Processing %s" % char)