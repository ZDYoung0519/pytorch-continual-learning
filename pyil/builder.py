from torchvision import transforms
from torchvision import datasets

from torch import optim
from torchvision.models import resnet18, resnet34, resnet50
from pyil.models import resnet32
from pyil.learner import EWC


def build_model(args):
    model_name = args.type
    args.pop('type')
    if model_name == 'resnet18':
        return resnet18(args)
    elif model_name == 'resnet34':
        return resnet34(args)
    elif model_name == 'resnet50':
        return resnet50(args)
    elif model_name == 'resnet32':
        return resnet32(args)


def build_transforms(dataset_name, flag='train'):
    if dataset_name == 'cifar100' and flag == 'train':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        ])
    return train_transforms, test_transforms


def build_dataset(args):
    assert args.train.type == args.test.type
    dataset_name = args.train.type
    train_transforms, test_transforms = build_transforms(dataset_name, 'train')
    if dataset_name == 'cifar10':
        train_dataset = datasets.cifar.CIFAR10(args.train.dataset_dir, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.cifar.CIFAR10(args.test.dataset_dir, train=False, download=True, transform=test_transforms)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.cifar.CIFAR100(args.train.dataset_dir, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.cifar.CIFAR100(args.test.dataset_dir, train=False, download=True, transform=test_transforms)
    elif dataset_name == 'mnist':
        train_dataset = datasets.mnist.MNIST(args.train.dataset_dir, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.mnist.MNIST(args.test.dataset_dir, train=True, download=True, transform=test_transforms)
    elif dataset_name == 'TinyImageNet':
        # TODO: Implement of tiny imagenet
        pass
    else:
        raise NotImplemented(f"Dataset: '{dataset_name}' not implemented!")
    return train_dataset, test_dataset


def build_leaner(args):
    learner = args['type']
    args.pop('type')
    if learner == 'ewc':
        return EWC(**args)

