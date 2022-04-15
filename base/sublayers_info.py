import torch

def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            #name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
            

    fs = ''
    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


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
    import torchvision.models as models
    model = models.resnet18()

    model_info(model, True)

    print("\n"*5+"="*30+"layers_info"+"="*30)
    layers_info(model)