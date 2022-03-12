import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms

imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# unused
transform_train = transforms.Compose([
    transforms.RandomCrop(imagesize, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# unused
transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_id_data(dataset: str, batch_size: int, **kwargs) -> DataLoader:
    # transform没写，之后看是否需要
    if dataset == "imagenet":
        valset = torchvision.datasets.ImageFolder(Path('data/imagenet/val'), transform_test_largescale)

    elif dataset == "CIFAR10":
        valset = torchvision.datasets.CIFAR10(Path('./data'), train=False,
                transform=transform_test, download=True)

    elif dataset == "CIFAR100":
        valset = torchvision.datasets.CIFAR100(Path('./data'), train=False,
                transform=transform_test, download=True)

    return DataLoader(valset, batch_size=batch_size,
            num_workers=1, **kwargs)

def load_ood_data(dataset: str, batch_size: int, **kwargs) -> DataLoader:

    if dataset == 'CIFAR-100':
        valset = torchvision.datasets.CIFAR100(Path('./data'), train=False,
                download=True, transform=transform_test)
    elif dataset == 'SVHN':
        valset = torchvision.datasets.ImageFolder(Path('./data/svhn/'),
                transform=transform_test)
    elif dataset == 'dtd':
        valset = torchvision.datasets.ImageFolder(Path("./data/dtd/images"),
                transform=transform_test_largescale)
    elif dataset == 'places50':
        valset = torchvision.datasets.ImageFolder(Path("./data/Places"),
                transform=transform_test_largescale)
    elif dataset == 'sun50':
        valset = torchvision.datasets.ImageFolder(Path("./data/SUN"),
                transform=transform_test_largescale)
    elif dataset == 'inat':
        valset = torchvision.datasets.ImageFolder(Path("./data/iNaturalist"),
                transform=transform_test_largescale)
    elif dataset == 'imagenet':
        valset = torchvision.datasets.ImageFolder(Path('./data/imagenet/val'),
                transform_test_largescale)
    else:
        valset = torchvision.datasets.ImageFolder(Path("./data") / dataset,
                transform=transform_test)

    return DataLoader(valset, batch_size=batch_size,
            num_workers=2, **kwargs)

