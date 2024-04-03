from torchvision import datasets, transforms


_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_CIFAR_TRAIN_TRANSFORMS = [
    transforms.ToTensor(),
]

_Flowers102_TRAIN_TRANSFORMS = [
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
]

_CIFAR_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]


_Flowers102_TEST_TRANSFORMS = [
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
]




TRAIN_DATASETS = {
    'mnist': datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'cifar10': datasets.CIFAR10(
        './datasets/cifar10', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'cifar100': datasets.CIFAR100(
        './datasets/cifar100', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'Flowers102': datasets.Flowers102( # Flowers102 has more datapoint in the test split
        './datasets/Flowers102',split = 'test', download=True,
        transform=transforms.Compose(_Flowers102_TRAIN_TRANSFORMS)
    )
}


TEST_DATASETS = {
    'mnist': datasets.MNIST(
        './datasets/mnist', train=False, download=True,
        transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
    ),
    'cifar10': datasets.CIFAR10(
        './datasets/cifar10', train=False, download=True,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'cifar100': datasets.CIFAR100(
        './datasets/cifar100', train=False, download=True,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'Flowers102': datasets.Flowers102(
        './datasets/Flowers102', split= 'train', download=True,    
        transform=transforms.Compose(_Flowers102_TRAIN_TRANSFORMS)
    )
}


DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'Flowers102': {'size': 64, 'channels': 3, 'classes': 102}
}
