import torch

from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda

# Use Nvidia CUDA if available
device = 'cpu' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(784, 30),
            nn.Sigmoid(),
            nn.Linear(30, 10),
            nn.Sigmoid()
        )

    def forward(self, data):
        data = self.flatten(data)
        logits = self.mlp(data)
        return logits


class MirrorMNIST(MNIST):
    # Original Dataset download is broken. Use mirror listed in https://github.com/cvdfoundation/mnist
    resources = [
        ("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
         "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
         "d53e105ee54ea40749a09fcbcd1e9432"),
        ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
         "9fb629c4189551a2d022fa330f9573f3"),
        ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
         "ec29112dd5afa0611ce80d1b7f02629c")
    ]


class Hyperparameters:
    LEARNING_RATE: float = 3
    EPOCHS: int = 10
    MINI_BATCH_SIZE: int = 10


def _train(model: TorchMLP, training_loader: DataLoader, learning_rate: float):
    optimizer = SGD(model.parameters(), learning_rate)
    loss_function = nn.MSELoss()

    for input, expected_output in training_loader.dataset:
        prediction = model(input.to(device))
        loss = loss_function(prediction, expected_output.to(device))

        # Backpropagation steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _test_accuracy(model: TorchMLP, testing_loader: DataLoader, epoch: int):
    total_size = len(testing_loader.dataset)
    correct_classifications = 0
    for input, expected_output in testing_loader.dataset:
        prediction = model(input.to(device))
        predicted_digit = prediction.to(device).argmax().__index__()
        expected_digit = expected_output.argmax().__index__()
        if expected_digit == predicted_digit:
            correct_classifications += 1
    print(f'Accuracy on testing data for epoch {epoch} is: {round((correct_classifications / total_size * 100), 2)}%')


def train_and_eval_torch_mlp():
    train_data = MirrorMNIST(root='mnist_torch_data', train=True, download=False, transform=ToTensor(),
                             target_transform=Lambda(
                                 lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y),
                                                                                       value=1).reshape(1, 10)
                             ))
    test_data = MirrorMNIST(root='mnist_torch_data', train=False, download=False, transform=ToTensor(),
                            target_transform=Lambda(
                                lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y),
                                                                                      value=1).reshape(1, 10)
                            ))

    params = Hyperparameters
    training_loader = DataLoader(train_data, batch_size=params.MINI_BATCH_SIZE, shuffle=True)
    testing_loader = DataLoader(test_data, batch_size=params.MINI_BATCH_SIZE, shuffle=True)

    model = TorchMLP().to(device)
    for epoch in range(params.EPOCHS):
        print(f"Training for epoch: {epoch}")
        _train(model, training_loader, params.LEARNING_RATE)
        _test_accuracy(model, testing_loader, epoch)


if __name__ == '__main__':
    train_and_eval_torch_mlp()
