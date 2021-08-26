import numpy as np

arr = np.array([[1, 2 , 4, 5 , 6], [1, 3, 4, 5, 6], [1, 2 , 4, 5 , 6]])
result2 = np.unique(arr,axis=0)
print(type(result2))  # <class 'numpy.ndarray'>
print(result2)  # [1 2 3]