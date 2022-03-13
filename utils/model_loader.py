from pathlib import Path

def load_model(model, dataset, filter, class_count):
    if dataset == 'imagenet':
        if model == 'resnet18':
            from models.resnet import resnet18
            model = resnet18
        elif model == 'resnet50':
            from models.resnet import resnet50
            model = resnet50
        elif model == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            model = mobilenet_v2

    else: # CIFAR
        if model == 'resnet18':
            from models.resnet import resnet18_cifar
            model = resnet18_cifar
        elif model == 'resnet50':
            from models.resnet import resnet50_cifar
            model = resnet50_cifar
        else:
            raise Exception("Model not supported")

    model = model(num_classes=class_count, filter=filter)
    model.eval()
    print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')

    return model
