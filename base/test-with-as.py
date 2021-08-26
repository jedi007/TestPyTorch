class FkResource:
    def __init__(self, tag):
        self.tag = tag
        print('构造器,初始化资源: %s' % tag)
    # 定义__enter__方法，with体之前的执行的方法
    def __enter__(self):
        print('[__enter__ %s]: ' % self.tag)
        # 该返回值将作为as子句中变量的值
        return 'fkit'  # 可以返回任意类型的值
    # 定义__exit__方法，with体之后的执行的方法
    def __exit__(self, exc_type, exc_value, exc_traceback):
        print('[__exit__ %s]: ' % self.tag)
        # exc_traceback为None，代表没有异常
        if exc_traceback is None:
            print('没有异常时关闭资源')
        else:
            print('遇到异常时关闭资源')
            return False   # 可以省略，默认返回None也被看做是False
with FkResource('孙悟空') as dr:
    print(dr)
    print('[with代码块] 没有异常')
print('------------------------------')
with FkResource('白骨精'):
    print('[with代码块] 异常之前的代码')
    raise Exception
    print('[with代码块] ~~~~~~~~异常之后的代码')


print('\n------------------------------2')
from contextlib import contextmanager
@contextmanager
def file_manager(name, mode):
    try:
        f = open(name, mode)
        yield f
    finally:
        f.close()
       
with file_manager('a.txt', 'w') as f:
    f.write('hello world')