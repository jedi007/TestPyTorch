import os
import json
from pprint import pprint

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from model import resnet18 as resnet

def main():
    current_dir = os.path.dirname(__file__)
    data_root = os.path.abspath(os.path.join(current_dir, "../.."))


    # create model
    model = resnet(num_classes=5)


    # load model weights
    weights_path = os.path.join(current_dir, "weights", "resNet18.pth")
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))


    # print(model.named_parameters)

    layers_info(model)

    print("model.named_parameters(): ", type(model.named_parameters()))

    # parameters_dict = dict()
    # for i, (name, p) in enumerate(model.named_parameters()):
    #     # print("name: ",name)
    #     # print("p: ",type(p))
    #     # print("p: ",p.size())
    #     # print("p: ",p.shape)
    #     # print("p: ",p.detach().numpy().shape )
    #     parameters_dict[name] = p.detach().numpy()
    
    # np.savez(".\\resnet_18.npz", resnet_dict = parameters_dict)  


    parameters_dict = dict()
    for i, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, torch.nn.Conv2d):
            layer_name = f'{name}.weight'
            p = m.weight
            parameters_dict[layer_name] = p.detach().numpy()

        elif isinstance(m, torch.nn.BatchNorm2d):
            layer_name = f'{name}.weight'
            p = m.weight
            parameters_dict[layer_name] = p.detach().numpy()

            layer_name = f'{name}.bias'
            p = m.bias
            parameters_dict[layer_name] = p.detach().numpy()

            layer_name = f'{name}.mean'
            p = m.running_mean
            parameters_dict[layer_name] = p.detach().numpy()

            layer_name = f'{name}.var'
            p = m.running_var
            parameters_dict[layer_name] = p.detach().numpy()
        
        elif isinstance(m, torch.nn.Linear):
            layer_name = f'{name}.weight'
            p = m.weight
            parameters_dict[layer_name] = p.detach().numpy().T

            layer_name = f'{name}.bias'
            p = m.bias
            parameters_dict[layer_name] = p.detach().numpy().T


    np.savez(".\\resnet_18.npz", resnet_dict = parameters_dict)  
    
    #print("parameters_dict: ",parameters_dict.keys())

    pred = model( torch.ones(1,3,224,224, dtype = torch.float32) )
    print("pred: ",pred)
    print(f"pred[0]: {pred[0][0]:.9}" )



def layers_info(model):
    # Plots a line-by-line description of a PyTorch layers
    n_p = [0]
    n_g = [0]
    layers_count = [0]

    def handle_one_parameters(layer_name, p):
        print('%5g %40s %9s %12.10g %20s %10.3g %10.3g' %
            (layers_count[0], layer_name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
        n_p[0] += p.numel()
        if p.requires_grad:
            n_g[0] += p.numel()
        layers_count[0] += 1

    print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, torch.nn.Conv2d):
            handle_one_parameters(f'{name}.weight', m.weight)
        elif isinstance(m, torch.nn.BatchNorm2d):
            handle_one_parameters(f'{name}.weight', m.weight)
            handle_one_parameters(f'{name}.bias', m.bias)
            handle_one_parameters(f'{name}.mean', m.running_mean)
            handle_one_parameters(f'{name}.var', m.running_var)
        elif isinstance(m, torch.nn.Linear):
            handle_one_parameters(f'{name}.weight', m.weight)
            handle_one_parameters(f'{name}.bias', m.bias)
            

    fs = ''
    print('Model Summary: %g layers, %12.10g parameters, %12.10g gradients%s' % (layers_count[0], n_p[0], n_g[0], fs))

if __name__ == "__main__":
    main()