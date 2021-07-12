import string

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from week6.pwd_dataset_manager.dataset_manager import DatasetFactory, get_dataset
from week6.utils import get_random_password_chunk

device = "cuda"


def convert_str_to_tensor(string_to_convert: str):
    size = len(string_to_convert)
    converted_tensor = torch.zeros(size, 1).long()
    for index, char in enumerate(string_to_convert):
        converted_tensor[index][0] = string.printable.index(char)
    return converted_tensor.to(device)


class GRU_RNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, no_of_hidden_layers: int, output_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        # (L, N, H_in)
        self.gru = nn.GRU(self.embedding_dim, hidden_size, num_layers=no_of_hidden_layers)
        self.embedding = nn.Embedding(len(string.printable), embedding_dim=self.embedding_dim)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden_state):
        input = self.embedding(input)
        reshaped_input = input.view(1, 1, self.embedding_dim)
        input, hidden_state = self.gru(reshaped_input, hidden_state)
        output = self.linear(input)
        return output, hidden_state


class PasswordGuesserUsingRNN:
    def __init__(self):
        self.input_size = 6
        self.hidden_size = 100
        self.no_of_hidden_layers = 2
        self.output_size = len(string.printable)
        self.embedding_dim = 6
        self.epochs = 100000
        self.eta = 0.005

    def train_and_evaluate(self):
        gru_model = GRU_RNN(self.embedding_dim, self.hidden_size, self.no_of_hidden_layers, self.output_size).to(device)

        dataset_klass = DatasetFactory().get("ClixSense")
        dataset = get_dataset(dataset_klass)

        optimizer = torch.optim.Adam(gru_model.parameters(), lr=self.eta)
        loss_fn = CrossEntropyLoss()
        for epoch in range(self.epochs):
            hidden_state = self._init_hidden()
            loss = 0
            input, target = get_random_password_chunk(dataset)
            input_tensor = convert_str_to_tensor(input)
            target_tensor = convert_str_to_tensor(target)

            gru_model.zero_grad()
            for input, expected in zip(input_tensor, target_tensor):
                output, hidden_state = gru_model(input, hidden_state)
                loss += loss_fn(output[-1], expected)

            loss.backward()
            optimizer.step()

            loss = loss.item() / len(input)
            if epoch % 100 == 0:
                print(f'At epoch: {epoch} with loss: {loss}')
                start = "h"
                prediction = self.eval(gru_model, start, 5)
                print(f"Prediction is {prediction} for start with '{start}'")

    def eval(self, gru_model, password_start, max_length):
        prediction = password_start
        start_tensor = convert_str_to_tensor(password_start)

        hidden_state = self._init_hidden()
        for char in start_tensor:
            _, hidden_state = gru_model(char, hidden_state)

        input = start_tensor[-1]
        for char_gen in range(max_length - len(password_start)):
            output, hidden_state = gru_model(input, hidden_state)

            # understand below; taken from the ref colab
            output_dist = output.data.view(-1).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            char_predicted = string.printable[top_i]
            prediction += char_predicted
            input = convert_str_to_tensor(char_predicted)

        return prediction

    def _init_hidden(self):
        # (D∗num_layers, N, Hout)
        return torch.zeros(self.no_of_hidden_layers, 1, self.hidden_size).to(device)


if __name__ == '__main__':
    PasswordGuesserUsingRNN().train_and_evaluate()
