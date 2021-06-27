import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import AlexNet
from torchvision.transforms import transforms

device = "cpu"


class AlexNetCNN(AlexNet):
    def train_model(self, training_loader: DataLoader, verbose: bool = True):
        optimizer = SGD(params=self.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        loss_function = CrossEntropyLoss()
        for input, labels in training_loader:
            prediction = self(input.to(device))
            labels = labels.to(device)
            loss = loss_function(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            self.evaluate(training_loader, verbose=verbose)

    def evaluate(self, data_loader: DataLoader, data_type: str = "training",
                 verbose: bool = True) -> float:
        correct_classifications = 0
        dataset_size = len(data_loader.dataset)
        with torch.no_grad():
            for input, labels in data_loader:
                prediction = self(input.to(device))
                labels = labels.to(device)
                correct_classifications += (prediction.argmax(1) == labels).type(torch.float).sum().item()
        accuracy = round((correct_classifications / dataset_size) * 100, 2)
        if verbose:
            print(f'Accuracy on {data_type} data {accuracy}%')
        return accuracy


def cifar10_alextnet():
    model = AlexNetCNN(10).to(device)
    # need input size of 63 for alexnet
    transform = transforms.Compose(
        [
            transforms.Resize((63, 63)),
            transforms.ToTensor()
        ])
    train_data = CIFAR10(root='cifar10', train=True, download=True, transform=transform)
    test_data = CIFAR10(root='cifar10', train=False, download=True, transform=transform)

    training_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    testing_loader = DataLoader(test_data, batch_size=4)

    accuracies = []
    for epoch in range(25):
        print(f"Training for epoch: {epoch}")
        model.train_model(training_loader)
        accuracies.append(model.evaluate(testing_loader, "testing"))
    print(max(accuracies))


def cifar100_alexnet():
    model = AlexNetCNN(100).to(device)
    # need input size of 63 for alexnet
    transform = transforms.Compose(
        [
            transforms.Resize((63, 63)),
            transforms.ToTensor()
        ])
    train_data = CIFAR100(root='cifar100', train=True, download=True, transform=transform)
    test_data = CIFAR100(root='cifar100', train=False, download=True, transform=transform)

    training_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    testing_loader = DataLoader(test_data, batch_size=4)

    accuracies = []
    for epoch in range(25):
        print(f"Training for epoch: {epoch}")
        model.train_model(training_loader)
        accuracies.append(model.evaluate(testing_loader, "testing"))
    print(max(accuracies))


if __name__ == '__main__':
    cifar10_alextnet()
