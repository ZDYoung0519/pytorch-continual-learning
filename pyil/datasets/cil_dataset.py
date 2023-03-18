import torch
from torch import nn
from torchvision import datasets, transforms


def get_dataset(dataset_name, train_dir, test_dir, transforms=None):
    if dataset_name == 'cifar10':
        train_dataset = datasets.cifar.CIFAR10(train_dir, train=True, download=True, transform=transforms)
        test_dataset = datasets.cifar.CIFAR10(test_dir, train=False, download=True)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.cifar.CIFAR100(train_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(test_dir, train=False, download=True)
    elif dataset_name == 'mnist':
        train_dataset = datasets.mnist.MNIST(train_dir, train=True, download=True)
        test_dataset = datasets.mnist.MNIST(test_dir, train=True, download=True)
    else:
        raise NotImplemented(f"Dataset: '{dataset_name}' not implemented!")

if __name__ == '__main__':
    print(1)
