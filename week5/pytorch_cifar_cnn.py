from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms

from week2.pytorch_mlp import device
from week5.mnist_cnn import TorchCNN


def run_cifar10():
    model = TorchCNN(
        loss_func="CrossEntropyLoss",
        optimizer="SGD",
        learning_rate=1e-2,
        lmda_wt_decay=1e-4,
        batch_size=10,
        training_size=50000,
        testing_size=10000,
        seed=21,
        momentum=0.9,
        nn_stack=[nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout(0.0),
                  nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout(0.0),
                  nn.Flatten(),
                  nn.Linear(40 * 5 * 5, 40 * 5 * 5 * 2),
                  nn.ReLU(),
                  nn.Dropout(0.0),
                  nn.Linear(40 * 5 * 5 * 2, 10),
                  nn.ReLU()]
    ).to(device)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = CIFAR10(root='cifar10', train=True, download=True, transform=transform)
    test_data = CIFAR10(root='cifar10', train=False, download=True, transform=transform)

    training_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    testing_loader = DataLoader(test_data, batch_size=10, shuffle=True)

    accuracies = []
    for epoch in range(20):
        print(f"Training for epoch: {epoch}")
        model.train_model(training_loader)
        accuracies.append(model.evaluate(testing_loader, model.testing_size, "testing"))
    print(max(accuracies))


def run_cifar100():
    model = TorchCNN(
        loss_func="CrossEntropyLoss",
        optimizer="SGD",
        learning_rate=1e-2,
        lmda_wt_decay=1e-4,
        batch_size=10,
        training_size=50000,
        testing_size=10000,
        seed=21,
        momentum=0.9,
        nn_stack=[nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout(0.0),
                  nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout(0.0),
                  nn.Flatten(),
                  nn.Linear(40 * 5 * 5, 40 * 5 * 5 * 2),
                  nn.ReLU(),
                  nn.Dropout(0.0),
                  nn.Linear(40 * 5 * 5 * 2, 100),
                  nn.ReLU()]
    ).to(device)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = CIFAR100(root='cifar100', train=True, download=True, transform=transform)
    test_data = CIFAR100(root='cifar100', train=False, download=True, transform=transform)

    training_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    testing_loader = DataLoader(test_data, batch_size=10, shuffle=True)

    accuracies = []
    for epoch in range(20):
        print(f"Training for epoch: {epoch}")
        model.train_model(training_loader)
        accuracies.append(model.evaluate(testing_loader, model.testing_size, "testing"))
    print(max(accuracies))


if __name__ == '__main__':
    run_cifar10()
