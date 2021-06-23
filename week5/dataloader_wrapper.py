from abc import abstractmethod
from typing import Type, List

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor, transforms

from week2.pytorch_mlp import MirrorMNIST


class DataLoaderWrapper:
    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_validation_splits(self) -> List[int]:
        pass


class GetDataLoaderFactory:
    def get(self, dataset: str) -> Type[DataLoaderWrapper]:
        if dataset == "MNIST":
            return MNISTLoader
        elif dataset == "CIFAR10":
            return CIFAR10Loader
        elif dataset == "CIFAR100":
            return CIFAR100Loader
        else:
            raise NotImplementedError(f"Dataset {dataset} not configured")


class MNISTLoader(DataLoaderWrapper):
    def get_data(self):
        train_data = MirrorMNIST(root='mnist_torch_data', train=True, download=True, transform=ToTensor())
        test_data = MirrorMNIST(root='mnist_torch_data', train=False, download=True, transform=ToTensor())
        return train_data, test_data

    def get_validation_splits(self) -> List[int]:
        return [50000, 10000]


cifar_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class CIFAR10Loader(DataLoaderWrapper):
    def get_data(self):
        train_data = CIFAR10(root='cifar10', train=True, download=True, transform=cifar_transform)
        test_data = CIFAR10(root='cifar10', train=False, download=True, transform=cifar_transform)
        return train_data, test_data

    def get_validation_splits(self) -> List[int]:
        return [40000, 10000]


class CIFAR100Loader(DataLoaderWrapper):
    def get_data(self):
        train_data = CIFAR100(root='cifar100', train=True, download=True, transform=cifar_transform)
        test_data = CIFAR100(root='cifar100', train=False, download=True, transform=cifar_transform)
        return train_data, test_data

    def get_validation_splits(self) -> List[int]:
        return [40000, 10000]
