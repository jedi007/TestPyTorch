import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v2 import MobileNetV2
from model_v3 import mobilenet_v3_small


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    current_dir = os.path.dirname(__file__)
    data_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # load image
    img_path = os.path.join(data_root, "data", "flower_data", "flower_photos", "tulips", "10791227_7168491604.jpg")
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = os.path.join(current_dir, "class_indices.json")
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    use_V2 = False
    if use_V2:
        # create model
        model = MobileNetV2(num_classes=5).to(device)
        # load model weights
        model_weight_path = os.path.join(current_dir, "weights", "save", "MobileNetV2.pth")
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
    else:
        model = mobilenet_v3_small(num_classes=5)
        model.to(device)

        # load model weights
        model_weight_path = os.path.join(current_dir, "weights", "save", "MobileNetV3.pth")
        model.load_state_dict(torch.load(model_weight_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
