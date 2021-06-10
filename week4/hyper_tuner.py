import itertools
import time
from typing import Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.transforms import ToTensor

from week2.pytorch_mlp import MirrorMNIST
from week4.mlp_pytorch import TorchMLP, device


class EvalData:
    def __init__(self):
        self.data = pd.DataFrame(
            columns=["size", "epochs", "hidden_act_function", "output_act_function", "loss_func",
                     "optimizer", "learning_rate", "weight_decay", "batch_size",
                     "testing_dataset_type", "training_size", "testing_size",
                     "p_to_be_zeroed", "dropout_on_input_layer",
                     "best_accuracy", "avg_accuracy",
                     "avg_time_taken"])

    def add_record(self, data_dict: Dict):
        self.data = self.data.append(data_dict, ignore_index=True)
        print("Added record to eval DF")

    def get(self) -> pd.DataFrame:
        # push "best_accuracy", "avg_accuracy", "avg_time_taken" cols to the front and sort by "avg_accuracy"
        curr_cols = self.data.columns.tolist()
        updated_order = curr_cols[-3:] + curr_cols[:-3]
        self.data = self.data[updated_order]
        return self.data.sort_values(by="avg_accuracy", ascending=False)


class TestingConfig:
    CONFIG = {
        "size": [[784, 30, 10]],  # 0
        "epochs": [20],  # 1
        "hidden_act_function": ["Sigmoid", "ReLU", "Tanh"],  # 2
        "output_act_function": ["Sigmoid", "ReLU", "Tanh"],  # 3
        "loss_func": ["MSELoss"],  # 4
        "optimizer": ["SGD"],  # 5
        "learning_rate": [0.01, 0.1, 1],  # 6
        "weight_decay": [0.0],  # 7
        "batch_size": [10],  # 8
        "testing_dataset_type": ["validation"],  # 9
        "training_size": [1000],  # 10
        "testing_size": [1000],  # 11
        "p_to_be_zeroed": [0.0],  # 12
        "dropout_on_input_layer": [False],  # 13
    }


class HyperTuner:
    def tune(self, config: Dict, verbose: bool = True) -> pd.DataFrame:
        eval_data = EvalData()
        train_data = MirrorMNIST(root='mnist_torch_data', train=True, download=False, transform=ToTensor())
        test_data = MirrorMNIST(root='mnist_torch_data', train=False, download=False, transform=ToTensor())

        all_combinations = itertools.product(*config.values())
        for combination in all_combinations:
            # reconstruct the dict using the combination
            combination_dict = {k: v for k, v in zip(config.keys(), combination)}
            accuracies = np.array([])
            time_consumed = np.array([])

            for seed in [28, 35, 42]:
                batch_size = combination_dict["batch_size"]

                torch.manual_seed(seed)
                training_subset, validation_subset = random_split(train_data, lengths=[50000, 10000])
                training_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=True)
                testing_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

                testing_dataset_type = combination_dict["testing_dataset_type"]
                testing_loader = validation_loader if testing_dataset_type == "validation" else testing_loader
                output_act_fn = combination_dict["output_act_function"]

                model = TorchMLP(
                    size=combination_dict["size"],
                    loss_func=combination_dict["loss_func"],
                    hidden_act_function=combination_dict["hidden_act_function"],
                    output_act_function=combination_dict["output_act_function"],
                    output_act_function_kwargs={"dim": 1} if output_act_fn == "Softmax" else {},
                    optimizer=combination_dict["optimizer"],
                    learning_rate=combination_dict["learning_rate"],
                    lmda_wt_decay=combination_dict["weight_decay"],
                    p_to_be_zeroed=combination_dict["p_to_be_zeroed"],
                    batch_size=batch_size,
                    training_size=combination_dict["training_size"],
                    testing_size=combination_dict["testing_size"],
                    seed=seed,
                    dropout_on_input=combination_dict["dropout_on_input_layer"]
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
                print(eval_data.get())
        return eval_data.get()


if __name__ == '__main__':
    eval_data = HyperTuner().tune(TestingConfig.CONFIG, True)
