# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import string

from week6.utils import convert_str_to_tensor

device = "cpu"
import math

import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


def train_and_eval():
    # this is taken from the pytorch notebook
    ntokens = len(string.printable)  # the size of vocabulary
    emsize = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    bptt = 200
    batch_size = 20
    epochs = 10
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.0001  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()  # Turn on the train mode
    total_loss = 0.

    with open("tiny_shake/input.txt", "r") as file:
        train_data = file.read()

    train_data = convert_str_to_tensor(train_data, device="cpu")
    train_data = batchify(train_data, batch_size)

    for epoch in range(epochs):
        src_mask = model.generate_square_subsequent_mask(bptt).to(device)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i, bptt)
            optimizer.zero_grad()
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            # to viz: column wise data
            # result = data[:, 10]
            # out = []
            # for item in result:
            #     out.append(string.printable[item])
            #
            # print("".join(out))
            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 5

            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / 200
                print(f"Epoch {epoch}, batch: {batch} /{len(train_data) // bptt}, loss: {cur_loss}")
                total_loss = 0
        evaluate(model, train_data, criterion, bptt, ntokens)


def evaluate(eval_model, data_source, criterion, bptt, ntokens):
    eval_model.eval()  # Turn on the evaluation mode
    # total_loss = 0.
    # src_mask = eval_model.generate_square_subsequent_mask(bptt).to(device)
    # with torch.no_grad():
    #     for i in range(0, data_source.size(0) - 1, bptt):
    #         data, targets = get_batch(data_source, i, bptt)
    #         if data.size(0) != bptt:
    #             src_mask = eval_model.generate_square_subsequent_mask(data.size(0)).to(device)
    #         output = eval_model(data, src_mask)
    #         output_flat = output.view(-1, ntokens)
    #         total_loss += len(data) * criterion(output_flat, targets).item()

            # print(output.size())
    test_tensor, _ = get_batch(data_source, 0, bptt)
    print(test_tensor.size())
    test_tensor = test_tensor[0]
    src_mask = eval_model.generate_square_subsequent_mask(len(test_tensor)).to(device)
    temp = eval_model(test_tensor, src_mask)
    output = eval_model(test_tensor, src_mask).view(-1, ntokens)
    print(output.size())
    out = []
    for i in range(20):
        for j in range(20):
            output_dist = temp[j, i, :].exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            char_predicted = string.printable[top_i]
            out.append(char_predicted)
    print("".join(out))
    # print(len(out))

    # print("eval", total_loss / (len(data_source) - 1))


if __name__ == '__main__':
    train_and_eval()
