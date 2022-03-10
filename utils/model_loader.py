from pathlib import Path

def load_model(model, dataset, class_count):
    if dataset == 'imagenet':
        if model == 'resnet18':
            from models.resnet import resnet18
            model = resnet18(num_classes=class_count, pretrained=True)
        elif model == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=class_count, pretrained=True)
        elif model == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            model = mobilenet_v2(num_classes=class_count, pretrained=True)

    else: # CIFAR
        if model == 'resnet18':
            from models.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=class_count, pretrained=True)
        elif model == 'resnet50':
            from models.resnet import resnet50_cifar
            model == resnet50_cifar(num_classes=class_count, pretrained=True)
        else:
            raise Exception("Model not supported")

    print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')

    return model
