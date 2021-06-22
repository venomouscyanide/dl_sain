import itertools
from typing import List

import torch

from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Use Nvidia CUDA if available
from week2.pytorch_mlp import MirrorMNIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class TorchCNN(nn.Module):
    def __init__(self, loss_func: str,
                 optimizer: str, learning_rate: float, lmda_wt_decay: float, batch_size: int,
                 momentum: float, nn_stack: List[nn.Module], training_size: int = 60000,
                 testing_size: int = 10000, seed: int = 42):
        super().__init__()
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lmda_wt_decay = lmda_wt_decay
        self.mlp = nn.Sequential(*nn_stack)
        self.momentum = momentum
        optimizer_params = self._get_optimizer_params()
        self.optimizer = getattr(torch.optim, self.optimizer)(**optimizer_params)
        self.loss_function = getattr(nn, self.loss_func)()
        self.batch_size = batch_size
        self.training_size = training_size
        self.testing_size = testing_size
        self.seed = seed

    def _get_optimizer_params(self):
        opt_params = {
            "params": self.parameters(),
            "lr": self.learning_rate,
            "weight_decay": self.lmda_wt_decay,
            "momentum": self.momentum,
        }
        return opt_params

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(data)
        return logits

    def train_model(self, training_loader: DataLoader, verbose: int = True):
        torch.manual_seed(self.seed)
        for input, labels in itertools.islice(training_loader, self.training_size // self.batch_size):
            prediction = self(input.to(device))
            labels = labels.to(device)
            if self.loss_function._get_name() == 'MSELoss':
                labels = torch.nn.functional.one_hot(labels, 10).float()
            loss = self.loss_function(prediction, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if verbose:
            self.evaluate(training_loader, self.training_size, verbose=verbose)

    def evaluate(self, data_loader: DataLoader, dataset_size: int, data_type: str = "training",
                 verbose: bool = True) -> float:
        correct_classifications = 0
        with torch.no_grad():
            torch.manual_seed(self.seed)
            for input, labels in itertools.islice(data_loader, dataset_size // self.batch_size):
                prediction = self(input.to(device))
                labels = labels.to(device)
                correct_classifications += (prediction.argmax(1) == labels).type(torch.float).sum().item()
        accuracy = round((correct_classifications / dataset_size) * 100, 2)
        if verbose:
            print(f'Accuracy on {data_type} data {accuracy}%')
        return accuracy


def run():
    model = TorchCNN(
        loss_func="CrossEntropyLoss",
        optimizer="SGD",
        learning_rate=1e-2,
        lmda_wt_decay=1e-4,
        batch_size=10,
        training_size=60000,
        testing_size=10000,
        seed=21,
        momentum=0.9,
        nn_stack=[nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout(0.0),
                  nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Dropout(0.0),
                  nn.Flatten(),
                  nn.Linear(40 * 4 * 4, 40 * 4 * 4 * 2),
                  nn.ReLU(),
                  nn.Dropout(0.0),
                  nn.Linear(40 * 4 * 4 * 2, 10),
                  nn.ReLU()]
    ).to(device)
    train_data = MirrorMNIST(root='mnist_torch_data', train=True, download=True, transform=ToTensor())
    test_data = MirrorMNIST(root='mnist_torch_data', train=False, download=True, transform=ToTensor())
    training_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    testing_loader = DataLoader(test_data, batch_size=10, shuffle=True)
    accuracies = []
    for epoch in range(20):
        print(f"Training for epoch: {epoch}")
        model.train_model(training_loader)
        accuracies.append(model.evaluate(testing_loader, model.testing_size, "testing"))
    print(max(accuracies))


if __name__ == '__main__':
    run()
