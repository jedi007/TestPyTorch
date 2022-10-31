#导入相应的包  
import scipy  
import scipy.cluster.hierarchy as sch  
from scipy.cluster.vq import vq,kmeans,whiten  
import numpy as np  
import matplotlib.pylab as plt  
  
  
#生成待聚类的数据点,这里生成了20个点,每个点4维:  
points=scipy.randn(20,4)    
  



# #1. 层次聚类  
# #生成点与点之间的距离矩阵,这里用的欧氏距离:  
# disMat = sch.distance.pdist(points,'euclidean')   
# #进行层次聚类:  
# Z=sch.linkage(disMat,method='average')   
# #将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png  
# P=sch.dendrogram(Z)  
# plt.savefig('plot_dendrogram.png')  
# #根据linkage matrix Z得到聚类结果:  
# cluster= sch.fcluster(Z, t=1, criterion='inconsistent')   
  
# print('Original cluster by hierarchy clustering:', cluster )
  




#2. k-means聚类  
#将原始数据做归一化处理  
data=whiten(points)  
  
#codebook, distortion = kmeans(obs, k_or_guess, iter=20, thresh=1e-05, check_finite=True) 
#输入obs是数据矩阵,行代表数据数目,列代表特征维度; k_or_guess表示聚类数目;iter表示循环次数,最终返回损失最小的那一次的聚类中心; 
#输出有两个,第一个是聚类中心(codebook),第二个是损失distortion,即聚类后各数据点到其聚类中心的距离的加和. 
#k-means最后输出的结果其实是两维的,第一维是聚类中心,第二维是损失distortion,我们在这里只取第一维,所以最后有个[0]  
centroid=kmeans(data, 3)[0]   
print("k-means:",centroid )
# centroid=kmeans(data,max(cluster))[0]    
  
#使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
#vq(obs, code_book, check_finite=True) 
#根据聚类中心将所有数据进行分类.obs为数据,code_book则是kmeans产生的聚类中心. 
#输出同样有两个:第一个是各个数据属于哪一类的label,第二个和kmeans的第二个输出是一样的,都是distortion  
label=vq(data,centroid)[0]   
  
print("Final clustering by k-means:",label )



# my kmeans demo
fe = np.array([[1.9,2.0],
                     [1.7,2.5],
                     [1.6,3.1],
                     [0.1,0.1],
                     [0.8,0.3],
                     [0.4,0.3],
                     [0.22,0.1],
                     [0.4, 0.3],
                     [0.4,0.5],
                     [1.8,1.9]])
fe *= 10
 
book = np.array((fe[0], fe[1]))
print(type(book))
print("book: \n",book)
 
codebook, distortion = kmeans(fe, book, iter=50) #第二个参数： k_or_guess : int or ndarray
# 可以写kmeans(wf,2)， 2表示两个质心，同时启用iter参数
print("codebook:", codebook)
print("distortion: ", distortion)
 
plt.scatter(fe[:,0], fe[:,1], c='g')
plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
plt.show()