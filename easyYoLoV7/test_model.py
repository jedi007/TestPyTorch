from models.yolo import Model
import torch

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    
    ckpt = torch.load("D:/testProject/TestPyTorch/easyYoLoV7/yolov7.pt", map_location=device)  # load checkpoint
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    
    model = Model()
    model.info(verbose=True)
    model.load_state_dict(state_dict, strict=False)  # load

    model.to(device)
    model.train()
    
    img = torch.ones(1, 3, 640, 640).to(device)
    pred = model(img)

    print("pred: ", pred)
    #... ... [ 5.21558e-02, -7.60717e-01, -9.02585e-01,  ..., -7.87223e+00, -8.90521e+00, -8.54759e+00]]]]], device='cuda:0', grad_fn=<CloneBackward0>)]
