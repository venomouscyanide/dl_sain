import copy
import itertools
import time
from typing import Dict

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.transforms import ToTensor

from week2.pytorch_mlp import MirrorMNIST
from week4.mlp_pytorch import device
from week5.mnist_cnn import TorchCNN


class EvalData:
    def __init__(self):
        self.data = pd.DataFrame(
            columns=["epochs", "nn_stack", "loss_func",
                     "optimizer", "learning_rate", "weight_decay", "batch_size", "momentum",
                     "testing_dataset_type", "training_size", "testing_size",
                     "best_accuracy", "avg_accuracy",
                     "avg_time_taken"])

    def add_record(self, data_dict: Dict):
        self.data = self.data.append(data_dict, ignore_index=True)
        print(f"Added record to eval DF. Total records so far: {self.data.shape[0]}")

    def get(self, rearrange: bool) -> pd.DataFrame:
        if rearrange:
            # push "best_accuracy", "avg_accuracy", "avg_time_taken" cols to the front and sort by "avg_accuracy"
            curr_cols = self.data.columns.tolist()
            updated_order = curr_cols[-3:] + curr_cols[:-3]
            self.data = self.data[updated_order]
        return self.data.sort_values(by="avg_accuracy", ascending=False)


class HyperTuner:
    def tune(self, config: Dict, verbose: bool = True) -> pd.DataFrame:
        eval_data = EvalData()
        train_data = MirrorMNIST(root='mnist_torch_data', train=True, download=True, transform=ToTensor())
        test_data = MirrorMNIST(root='mnist_torch_data', train=False, download=True, transform=ToTensor())

        all_combinations = list(itertools.product(*config.values()))
        print(f"Total combinations for exp: {len(all_combinations)}")
        for combination in all_combinations:
            # reconstruct the dict using the combination
            combination_dict = {k: v for k, v in zip(config.keys(), combination)}
            accuracies = np.array([])
            time_consumed = np.array([])

            for seed in [28, 35, 42]:
                batch_size = combination_dict["batch_size"]
                testing_dataset_type = combination_dict["testing_dataset_type"]

                torch.manual_seed(seed)
                training_subset = train_data
                training_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
                testing_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

                if testing_dataset_type == "validation":
                    training_subset, validation_subset = random_split(train_data, lengths=[50000, 10000])
                    validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=True)
                    testing_loader = validation_loader if testing_dataset_type == "validation" else testing_loader

                model = TorchCNN(
                    loss_func=combination_dict["loss_func"],
                    optimizer=combination_dict["optimizer"],
                    learning_rate=combination_dict["learning_rate"],
                    lmda_wt_decay=combination_dict["weight_decay"],
                    batch_size=batch_size,
                    momentum=combination_dict["momentum"],
                    training_size=combination_dict["training_size"],
                    testing_size=combination_dict["testing_size"],
                    seed=seed,
                    nn_stack=combination_dict['nn_stack']
                ).to(device)

                num_epochs = combination_dict["epochs"]
                time_epoch_start = time.time()
                for epoch in range(num_epochs):
                    if verbose:
                        print(f"Training for epoch: {epoch}")
                    model.train_model(training_loader, verbose)
                    accuracy = model.evaluate(testing_loader, model.testing_size, "testing", verbose)
                    accuracies = np.append(accuracies, accuracy)
                time_for_seed = time.time() - time_epoch_start
                time_consumed = np.append(time_consumed, time_for_seed)

            avg_time = np.mean(time_consumed)
            avg_accuracy = np.mean(accuracies)
            best_accuracy = np.max(accuracies)

            combination_dict.update({
                "best_accuracy": best_accuracy,
                "avg_accuracy": avg_accuracy,
                "avg_time_taken": avg_time
            })

            eval_data.add_record(combination_dict)

            if verbose:
                print(eval_data.get(rearrange=False))
        return eval_data.get(rearrange=True)


def _get_nn_stacks_with_dropouts(base_architectures, dropout_options):
    # 3 loops poggers
    archs_with_dropout = []
    for base_architecture in base_architectures:
        for dropout_option in dropout_options:
            arch_copy = copy.deepcopy(base_architecture)
            for index, module in enumerate(arch_copy):
                if type(module) == nn.Dropout:
                    arch_copy[index] = nn.Dropout(dropout_option)
            archs_with_dropout += [arch_copy]
    return archs_with_dropout


if __name__ == '__main__':
    class TestingConfig:
        base_architectures = [[nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1),
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
                               nn.ReLU()]]
        CONFIG = {
            "epochs": [20],
            "nn_stack": _get_nn_stacks_with_dropouts(base_architectures, dropout_options=[0.0, 0.10, 0.20]),
            "loss_func": ["CrossEntropyLoss"],
            "optimizer": ["SGD"],
            "learning_rate": [1e-2],
            "weight_decay": [1e-4],
            "batch_size": [10],
            "testing_dataset_type": ["validation"],
            "training_size": [5000],
            "testing_size": [1000],
            "momentum": [0.9]
        }


    eval_data = HyperTuner().tune(TestingConfig.CONFIG, False)
