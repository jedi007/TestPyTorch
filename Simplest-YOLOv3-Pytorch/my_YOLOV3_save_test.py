import os
import json

import torch

from samplest_yolov3 import YOLOv3Model

def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    basepath = "D:/pythonproject/Detection/UPUP/deep-learning-for-image-processing-master/pytorch_object_detection/yolov3_spp"
    weights = basepath+"/weights/yolov3spp-voc-512.pt"  # 改成自己训练好的权重文件
    json_path = basepath+"/data/pascal_voc_classes.json"  # json标签文件
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("device: ",device)

    model = YOLOv3Model()
    model.loadPublicPt(weights,device)
    model.to(device)
    model.eval()

    img_size = 512
    input_size = (img_size, img_size)

    img = torch.ones((1, 3, img_size, img_size), device=device)

    net = torch.jit.trace(model, img)
    net.save('D:/TestData/my_yolov3_jit_cuda3.pt')
    # pred=net(img)
    
    # print( "pred.shape: ", pred.shape )
    # print( "pred[0][0]: ", pred[0][0] )
    # print( "pred[0][768]: ", pred[0][768] )
    # print( "pred[0][3840]: ", pred[0][3840] )


if __name__ == "__main__":
    main()
