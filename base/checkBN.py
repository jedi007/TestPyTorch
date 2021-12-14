import torch
import torch.nn as nn

def checkBN(debug = False):
    # parameters
    N = 5 # batch size
    C = 3 # channel
    W = 2 # width of feature map
    H = 2 # height of feature map
    # batch normalization layer
    BN = nn.BatchNorm2d(C,affine=True) #gamma和beta, 其维度与channel数相同
    # input and output
    featuremaps = torch.randn(N,C,W,H)
    output = BN(featuremaps)
    # checkout
    ###########################################
    if debug:
        print("input feature maps：\n",featuremaps)
        print("normalized feature maps: \n",output)
    ###########################################
    
    # manually operation, the first channel
    X = featuremaps[:,0,:,:]
    firstDimenMean = torch.Tensor.mean(X)
    firstDimenVar = torch.Tensor.var(X,False) #Bessel's Correction贝塞尔校正不被使用
    
    BN_one = ((featuremaps[0][0] - firstDimenMean)/(torch.pow(firstDimenVar+BN.eps,0.5) )) * BN.weight[0] + BN.bias[0]
    print('+++'*15,'\n','manually operation: ', BN_one)
    print('==='*15,'\n','pytorch result: ', output[0][0])
    
if __name__=="__main__":
    checkBN()
# ————————————————
# 版权声明：本文为CSDN博主「白水煮蝎子」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_44278406/article/details/105554268