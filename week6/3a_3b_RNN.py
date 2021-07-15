import argparse
import math
import os
import string
from collections import Counter
from math import ceil
from typing import List, Set, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from week6.pwd_dataset_manager.dataset_manager import DatasetFactory, get_dataset
from week6.utils import get_input_expected_clixsense, convert_str_to_tensor

device = "cuda"


class GRU_RNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, no_of_hidden_layers: int, output_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        # (L, N, H_in)
        self.gru = nn.GRU(self.embedding_dim, hidden_size, num_layers=no_of_hidden_layers)
        self.embedding = nn.Embedding(len(string.printable), embedding_dim=self.embedding_dim)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor):
        input = self.embedding(input)
        reshaped_input = input.view(1, 1, self.embedding_dim)
        input, hidden_state = self.gru(reshaped_input, hidden_state)
        output = self.linear(input)
        return output, hidden_state


class PasswordGuesserUsingRNN:
    def __init__(self):
        self.hidden_size = 100
        self.no_of_hidden_layers = 20
        self.output_size = len(string.printable)
        self.embedding_dim = 5
        self.epochs = 10
        self.eta = 1e-4

    def train_and_evaluate(self):
        dataset_klass = DatasetFactory().get("ClixSense")
        dataset = get_dataset(dataset_klass)

        gru_model = GRU_RNN(self.embedding_dim, self.hidden_size, self.no_of_hidden_layers, self.output_size).to(device)
        optimizer = torch.optim.Adam(gru_model.parameters(), lr=self.eta)
        loss_fn = CrossEntropyLoss()

        n_most_common, pwd_inp_exp = get_input_expected_clixsense(dataset)
        # total_len of dict = 1338980
        # total length of passwords = 2221027
        for epoch in range(self.epochs):
            for pwd_index, (most_common_pwd, num_occ) in enumerate(n_most_common[:]):
                inp_target_set = pwd_inp_exp[most_common_pwd]
                for _ in range(ceil((num_occ / 100000) * 100)):
                    for input_pwd, target_pwd in inp_target_set:
                        loss = 0
                        optimizer.zero_grad()

                        hidden_state = self._init_hidden()
                        input_tensor = convert_str_to_tensor(input_pwd)
                        target_tensor = convert_str_to_tensor(target_pwd)

                        for input, expected in zip(input_tensor, target_tensor):
                            output, hidden_state = gru_model(input, hidden_state)
                            loss += loss_fn(output[-1], expected)

                        loss.backward()
                        optimizer.step()

                        loss = loss.item() / len(input_pwd)

                if pwd_index % 1000 == 0:
                    print(f"At pwd_index: {pwd_index} of {len(n_most_common)}")
                    print(f"training password: {most_common_pwd}")

                    print(f'At epoch: {epoch} with loss: {loss}')
                    start = "123"
                    prediction = self.evaluate_password(gru_model, start, 15)
                    print(f"Prediction is {prediction} for start with '{start}'")

        return gru_model

    def evaluate_password(self, gru_model: nn.Module, password_start: str, max_length: int):
        prediction = password_start
        start_tensor = convert_str_to_tensor(password_start)
        with torch.no_grad():
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


class MakePasswordGuesses:
    MIN_LENGTH: int = 5
    MAX_LENGTH: int = 15
    MAX_TRIES_PER_CONFIG: int = 5
    MAX_GUESSES: int = int(math.pow(10, 6))

    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose

    def evaluate_dataset(self, dataset_name: str) -> Tuple[Set[str], Set[str]]:
        dataset_klass = DatasetFactory().get(dataset_name)
        dataset = get_dataset(dataset_klass)

        dataset_counter = Counter(dataset)
        dataset_len = len(dataset)
        print(f'Total passwords in {dataset_name} is {dataset_len}')
        most_common = dataset_counter.most_common()

        total_correct_guesses, guessed_passwords, all_starters_used = self._make_guesses(most_common, dataset_counter,
                                                                                         dataset_len)
        missed_passwords = set(dataset_counter.keys()).difference(guessed_passwords)

        print(
            f"Unique guesses correct: {len(guessed_passwords)} and Total guesses: {total_correct_guesses} and total misses: {len(missed_passwords)}"
        )
        print(f"Coverage on {dataset_name}: {(total_correct_guesses / len(dataset)) * 100}")

        self._write_debug(dataset_name, "all_starters.txt", all_starters_used)
        self._write_debug(dataset_name, "unique_correct_guesses.txt", guessed_passwords)
        self._write_debug(dataset_name, "missed_passwords.txt", missed_passwords)

        return guessed_passwords, missed_passwords

    def _form_candidates(self, common_pwd: str) -> List[str]:
        min_len = 3
        return [common_pwd[0:end] for end in range(min_len, len(common_pwd))]

    def _make_guesses(self, most_common: List[Tuple[str, int]], dataset_counter: Counter, dataset_len: int):
        # TODO: Refactor. Super messy right now.
        importance = 4
        total_correct_guesses = 0
        uniq_guessed_passwords = set()
        all_starters_used = set()
        total_guess_tracker = 0

        seen_edge_cases = set()
        guesser = PasswordGuesserUsingRNN()
        for common_pwd, common_occ in most_common:
            starter_candidates = self._form_candidates(common_pwd)
            all_starters_used |= set(starter_candidates)

            for candidate in starter_candidates:
                if len(candidate) > self.MAX_LENGTH:
                    if candidate not in seen_edge_cases:
                        seen_edge_cases.add(candidate)
                        if total_guess_tracker > self.MAX_GUESSES:
                            return total_correct_guesses, uniq_guessed_passwords, all_starters_used

                        total_guess_tracker += 1
                        guess = candidate
                        total_correct_guesses = self._update_if_guess_is_correct(uniq_guessed_passwords, guess,
                                                                                 candidate, max_len,
                                                                                 dataset_counter, total_correct_guesses)
                    continue

                for max_len in range(self.MIN_LENGTH, self.MAX_LENGTH + 1):

                    if max_len < len(candidate):
                        continue

                    if max_len == len(candidate):
                        if candidate not in seen_edge_cases:
                            seen_edge_cases.add(candidate)
                            if total_guess_tracker > self.MAX_GUESSES:
                                return total_correct_guesses, uniq_guessed_passwords, all_starters_used

                            total_guess_tracker += 1
                            guess = candidate
                            total_correct_guesses = self._update_if_guess_is_correct(uniq_guessed_passwords, guess,
                                                                                     candidate, max_len,
                                                                                     dataset_counter,
                                                                                     total_correct_guesses)
                        continue

                    for _ in range(self.MAX_TRIES_PER_CONFIG * ceil(common_occ / (dataset_len / importance) * 100)):
                        if total_guess_tracker > self.MAX_GUESSES:
                            return total_correct_guesses, uniq_guessed_passwords, all_starters_used

                        total_guess_tracker += 1
                        if total_guess_tracker % 1000 == 0 and self.verbose:
                            print(f"At guess {total_guess_tracker} of {self.MAX_GUESSES}")

                        guess = guesser.evaluate_password(self.model, candidate, max_length=max_len)
                        total_correct_guesses = self._update_if_guess_is_correct(uniq_guessed_passwords, guess,
                                                                                 candidate, max_len,
                                                                                 dataset_counter, total_correct_guesses)

        return total_correct_guesses, uniq_guessed_passwords, all_starters_used

    def _update_if_guess_is_correct(self, uniq_guessed_passwords: Set[str], guess: str, candidate: str, max_len: int,
                                    dataset_counter: Counter, total_correct_guesses: int):
        if guess not in uniq_guessed_passwords:
            occurrences = dataset_counter.get(guess)
            if occurrences:
                if self.verbose:
                    print(
                        f"Correct guess: {guess}, for candidate: {candidate}, given max_len: {max_len} \nTotal Correct guesses so far:{total_correct_guesses}"
                    )
                total_correct_guesses += occurrences
                uniq_guessed_passwords.add(guess)
        return total_correct_guesses

    def _write_debug(self, dataset_name: str, file_name: str, data_as_set: Set[str]):
        debug_folder = f'debug_{dataset_name}'
        if not os.path.exists(debug_folder):
            os.mkdir(debug_folder)

        data_as_str = '\n'.join(data_as_set)
        with open(os.path.join(debug_folder, file_name), "w") as debug_file:
            debug_file.write(data_as_str)


def train_and_save():
    gru_model = PasswordGuesserUsingRNN().train_and_evaluate()
    print(PasswordGuesserUsingRNN().evaluate_password(gru_model, "123", 20))
    torch.save(gru_model, 'saved_gru_pwd.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_name', required=True,
                        help='The name of the dataset on which guessing is to be conducted on',
                        choices=['Mate1', '000webhost'])

    args = parser.parse_args()
    dataset_name = args.dataset_name
    print(f"Running guessing on: {dataset_name}")
    gru_model = torch.load('saved_gru_pwd.model')
    MakePasswordGuesses(gru_model).evaluate_dataset(dataset_name)
