import  os
print(os.getcwd()) # 获取当前工作目录路径
print(os.path.abspath('.')) # 获取当前工作目录路径

current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录

print("current_work_dir: ",current_work_dir)
 
weight_path = "D:/"
weight_path = os.path.join(current_work_dir, weight_path)  # 再加上它的相对路径，这样可以动态生成绝对路径
print(weight_path)