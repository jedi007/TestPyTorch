from models import yolox
from models import darknet

if __name__ == "__main__":
    model = yolox.YOLOX() #默认相当于yolox-l
    print("yolox: \n",model)

    # model_CSPDarknet = darknet.CSPDarknet(dep_mul=1.0,wid_mul=1.0)
    # print("model_CSPDarknet: \n",model_CSPDarknet)