# Shakespeare trials using GRU and other RNNs
# source is Section 2 in https://colab.research.google.com/github/lcharlin/80-629/blob/master/week6-RNNs%2BCNNs/RNNs_Answers.ipynb#scrollTo=49GUzizBbllV
# With some comments from me to better understanding + moving the tensors and model to GPU
# The above colab mentions this link: https://colab.research.google.com/drive/1jR_DGoVDcxZ104onxTk2C7YeV7vTt1DV#scrollTo=2bJvq3okegMw


# Docs for GRU: https://pytorch.org/docs/master/generated/torch.nn.GRU.html#torch.nn.GRU
# Help understand the dimensionality of input/output/hidden layers explained by referring the above link: https://stackoverflow.com/a/45023288

import string

import torch

import random
import unidecode
from torch import nn
import torch.nn.functional as F

file = unidecode.unidecode(open('tiny_shake/input.txt').read())

device = "cuda"


def random_chunk(chunk_len=200):
    # randomly sample a chunk of 200 characters for training from input.txt
    start_index = random.randint(0, len(file) - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]


def char_tensor(str, print_=False):
    # convert each char to a index given by string package
    # and store that value in the appropriate index in the tensor of floats
    tensor = torch.zeros(len(str), 1).long()
    for c in range(len(str)):
        tensor[c][0] = string.printable.index(str[c])
        if print_:
            print(f"Character: {str[c]} gets index {tensor[c][0]}")
    return tensor.to(device)


def random_train_data():
    # this makes sense. input is at time t, and target is at time t+1
    chunk = random_chunk()
    input = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return input, target


class ShakespeareRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(ShakespeareRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embeding_size = 6

        self.embedding = nn.Embedding(len(string.printable), self.embeding_size)
        # GRU args are (6, 100, 2)
        # args to GRU chosen below (num_features, hidden_size, number_of_layers to stack)
        self.rnn = nn.GRU(self.embeding_size, hidden_size, num_layers=n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size).to(device)

    def forward(self, input, hidden):
        x = self.embedding(input)
        # The input dimensions are (seq_len, batch, input_size)
        # https://pytorch.org/docs/stable/tensor_view.html
        # x is (1,6)
        # we have to make x => (1, 1, 6)
        x, hidden = self.rnn(x.view(1, 1, x.size(-1)), hidden)
        output = self.decoder(x)
        # output is (1, 1, 100) as self.output_size = 100
        # TODO: what is the use of variable `tag_scores`
        tag_scores = F.log_softmax(output[-1], dim=1)
        return output, hidden


def evaluate(model, prime_str='A', predict_len=100):
    hidden = model.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)
    input = prime_input[-1]

    for p in range(predict_len):
        output, hidden = model(input, hidden)

        # Sample from the network as a multinomial distribution
        # TODO: Why are we doing this? Did not understand so far.
        output_dist = output.data.view(-1).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = string.printable[top_i]
        predicted += predicted_char
        input = char_tensor(predicted_char)

    return predicted


def train_and_eval_shakespeare():
    n_characters = len(string.printable)
    hidden_size = 100
    n_layers = 2

    epochs = 2000
    lr = 0.005

    model = ShakespeareRNN(n_characters, hidden_size, n_characters, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = 0
        input, target = random_train_data()
        chunk_len = len(input)
        hidden = model.init_hidden()
        # the hidden dimensions are (num_layers, batch, hidden_size)
        # hidden.size = torch.Size([2, 1, 100])

        model.zero_grad()
        for x, y in zip(input, target):
            out, hidden = model(x, hidden)
            loss += criterion(out[-1], y)

        loss.backward()
        optimizer.step()

        loss = loss.item() / chunk_len
        if epoch % 100 == 0:
            print('[(%d %d%%) %.4f]' % (epoch, epoch / epochs * 100, loss))
            print(evaluate(model, 'Wh', 100), '\n')
    return model


if __name__ == '__main__':
    char_tensor('ASCII', print_=True)
    model = train_and_eval_shakespeare()
    print(evaluate(model, 'Poll:', 100), '\n')
