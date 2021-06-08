import itertools
from typing import List, Any, Dict

import torch

from torch import nn

from torch.utils.data import DataLoader, random_split

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from week2.pytorch_mlp import MirrorMNIST

# Use Nvidia CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class TorchMLP(nn.Module):
    def __init__(self, size: List[int], loss_func: str, hidden_act_function: str, output_act_function: str,
                 output_act_function_kwargs: Dict[Any, Any],
                 optimizer: str, learning_rate: float, lmda_wt_decay: float, p_to_be_zeroed: float, batch_size: int,
                 training_size: int = 60000,
                 testing_size: int = 10000, seed: int = 42):
        super().__init__()
        self.size = size
        self.loss_func = loss_func
        self.hidden_act_function = hidden_act_function
        self.output_act_function = output_act_function
        self.output_act_function_kwargs= output_act_function_kwargs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lmda_wt_decay = lmda_wt_decay
        self.p_to_be_zeroed = p_to_be_zeroed
        self.flatten = nn.Flatten()
        nn_stack = self._form_nn_stack()
        self.mlp = nn.Sequential(*nn_stack)
        optimizer_params = self._get_optimizer_params()
        self.optimizer = getattr(torch.optim, self.optimizer)(**optimizer_params)
        self.loss_function = getattr(nn, self.loss_func)()
        self.batch_size = batch_size
        self.training_size = training_size
        self.testing_size = testing_size
        self.seed = seed

    def _form_nn_stack(self):
        nn_stack = []
        # hidden layers
        for layer in range(len(self.size) - 2):
            nn_stack.append(nn.Linear(self.size[layer], self.size[layer + 1]))
            nn_stack.append(getattr(nn, self.hidden_act_function)())
            nn_stack.append(nn.Dropout(self.p_to_be_zeroed))
        # output layer
        nn_stack.append(nn.Linear(self.size[-2], self.size[-1]))
        nn_stack.append(getattr(nn, self.output_act_function)(**self.output_act_function_kwargs))
        return nn_stack

    def _get_optimizer_params(self):
        opt_params = {
            "params": self.parameters(),
            "lr": self.learning_rate,
            "weight_decay": self.lmda_wt_decay
        }
        return opt_params

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self.flatten(data)
        logits = self.mlp(data)
        return logits

    def train_model(self, training_loader: DataLoader):
        torch.manual_seed(self.seed)
        for input, labels in itertools.islice(training_loader, self.training_size // self.batch_size):
            prediction = self(input.to(device))
            labels = labels.to(device)
            loss = self.loss_function(prediction, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.evaluate(training_loader, self.training_size, "testing")

    def evaluate(self, data_loader: DataLoader, dataset_size: int, data_type: str = "training"):
        correct_classifications = 0
        with torch.no_grad():
            torch.manual_seed(self.seed)
            for input, labels in itertools.islice(data_loader, dataset_size // self.batch_size):
                prediction = self(input.to(device))
                labels = labels.to(device)
                correct_classifications += (prediction.argmax(1) == labels).type(torch.float).sum().item()
        print(f'Accuracy on {data_type} data {round((correct_classifications / dataset_size) * 100, 2)}%')


def one_hot_encode(y):
    return torch.zeros(10, dtype=torch.long).scatter_(0, torch.tensor(y), value=1).reshape(1, 10)


def run():
    # TODO: MSELoss() option is broken
    train_data = MirrorMNIST(root='mnist_torch_data', train=True, download=False, transform=ToTensor())
    test_data = MirrorMNIST(root='mnist_torch_data', train=False, download=False, transform=ToTensor())

    torch.manual_seed(42)
    training_subset, validation_subset = random_split(train_data, lengths=[50000, 10000])
    training_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    validation_loader = DataLoader(validation_subset, batch_size=10, shuffle=True)
    testing_loader = DataLoader(test_data, batch_size=10, shuffle=True)

    model = TorchMLP(
        size=[784, 30, 10],
        loss_func="CrossEntropyLoss",
        hidden_act_function="Sigmoid",
        output_act_function="Softmax",
        output_act_function_kwargs={"dim": 1},
        optimizer="SGD",
        learning_rate=0.1,
        lmda_wt_decay=0.0,
        p_to_be_zeroed=0.20,
        batch_size=10,
        training_size=5000,
        testing_size=2000
    ).to(device)
    for epoch in range(10):
        print(f"Training for epoch: {epoch}")
        model.train_model(training_loader)
        model.evaluate(testing_loader, model.testing_size)


if __name__ == "__main__":
    run()
