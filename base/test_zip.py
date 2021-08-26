import numpy as np
import torch

# 定义三个列表
a = [1, 2, 3]
b = [4, 5, 6]
c = [4, 5, 6, 7, 8]

print(*a)

 # 打包为元组的列表,而且元素个数与最短的列表一致
a_b = zip(a, b)
# 输出zip函数的返回对象类型
print("a_b类型%s" % type(a_b))
# 输出a_b
print(a_b)

print(list(a_b))

print("step======================================================1")


# 声明一个列表
nums = [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3']]

print(*nums)

# 参数为list数组时，是压缩数据，相当于zip()函数
iters = zip(*nums)  
# 输出zip(*zipped)函数返回对象的类型
print("type of iters is %s" % type(iters))  
# 因为zip(*zipped)函数返回一个zip类型对象，所以我们需要对其进行转换
# 在这里，我们将其转换为字典
print(dict(iters))



print("step======================================================2")
# 创建2个列表
m = [1, 2, 3]
n = [4, 5, 6]

print("*zip(m, n)返回:", *zip(m, n))
m2, n2 = zip(*zip(m, n))
print("m2和n2的值分别为:", m2, n2)
# 若相等，返回True；说明*zip为zip的逆过程
print(m == list(m2) and n == list(n2))

print("*zip(*zip(m, n)):", *zip(*zip(m, n)) )




