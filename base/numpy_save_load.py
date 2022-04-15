import numpy as np


# 非压缩存储： 以二进制的方式存储文件，在二进制文件第一行以文本形式保存了数据的元信息（ndim，dtype，shape等），可以用二进制工具查看内容。
outfile = r'.\\test.npy'
np.random.seed(20200619)
x = np.random.uniform(low=0, high=1,size = [3, 5])
print("x: ", x)
np.save(outfile, x)
y = np.load(outfile)
print("y: ", y)



print("\n"*5+"="*30+"压缩存储"+"="*30)
# 压缩存储： def savez(file, *args, **kwds):  可以将多个数组保存到一个文件
# savez()函数：以未压缩的.npz格式将多个数组保存到单个文件中。
# .npz格式：以压缩打包的方式存储文件，可以用压缩软件解压。
# savez()函数：第一个参数是文件名，其后的参数都是需要保存的数组，也可以使用关键字参数为数组起一个名字，非关键字参数传递的数组会自动起名为arr_0, arr_1, …。
# savez()函数：输出的是一个压缩文件（扩展名为.npz），其中每个文件都是一个save()保存的.npy文件，文件名对应于数组名。load()自动识别.npz文件，并且返回一个类似于字典的对象，
# 可以通过数组名作为关键字获取数组的内容。
outfile = r'.\\test2.npz'
x = np.linspace(0, np.pi, 5)
print("x: ",x)
y = np.sin(x)
print("y: ",y)
z = np.cos(x)
print("z: ",z)
np.savez(outfile, x, y, z_lj=z)  # z_lj 就是自己起的名
data = np.load(outfile)
np.set_printoptions(suppress=True)
print(data.files)  
# ['z_lj', 'arr_0', 'arr_1']

print(data['arr_0'])
# [0.         0.78539816 1.57079633 2.35619449 3.14159265]

print(data['arr_1'])
# [0.         0.70710678 1.         0.70710678 0.        ]

print(data['z_lj'])
# [ 1.          0.70710678  0.         -0.70710678 -1.        ]






print("\n"*5+"="*30+"压缩存储2"+"="*30)
outfile = r'.\\test3.npz'
x = np.random.uniform(low=0, high=1,size = [3, 5])
print("x: ",x)
y = np.random.uniform(low=0, high=1,size = [2, 3])
print("y: ",y)
z = np.random.uniform(low=0, high=1,size = [2, 5])
print("z: ",z)

save_z = dict()
save_z["lj1"] = x
save_z["lj2"] = y
save_z["lj3"] = z
print("save_z: ", save_z)
print("save_z type: ",type(save_z))

np.savez(outfile, resnet_dict = save_z)  
data = np.load(outfile, allow_pickle=True)
print("data: ",type(data))

np.set_printoptions(suppress=True)
print(data.files)
print(data['resnet_dict'])

print("dict = ",type(data['resnet_dict'].item()))





