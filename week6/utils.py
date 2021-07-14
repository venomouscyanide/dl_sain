import string
from collections import defaultdict, Counter
from random import randint
from typing import List, Tuple, Dict, Set

# min size of ClixSense is 6
# max size of ClixSense is 25
# most common with their no_of_occurrences
# [('123456', 17871), ('123456789', 3294), ('12345678', 2091), ('password', 1967), ('111111', 1892),
# ('1234567', 1299), ('iloveyou', 1266), ('qwerty', 1187), ('clixsense', 1172), ('000000', 977)]
import torch

from week6.trials_on_RNN import device


def get_random_password_chunk(dataset: List[str]) -> Tuple[str, str]:
    total_len = len(dataset)
    random_password = dataset[randint(0, total_len - 1)]

    len_password = len(random_password)
    random_cut = randint(1, len_password - 1 - 1)

    inp = random_password[0:len_password - 1]
    expected = random_password[0:len_password]
    return inp, expected


def get_input_expected_clixsense(dataset: List[str]) -> Tuple[List[Tuple[str, int]], Dict[str, Set[Tuple[str, str]]]]:
    password_slices_dict = defaultdict(set)
    [password_slices_dict[pwd].add((pwd[0: -1], pwd[1:])) for pwd in dataset[:]]

    # cut_off = 4
    # [password_slices_dict[pwd].add((pwd[index: len(pwd)], pwd[index + 1: len(pwd)] + '')) for pwd in dataset[:10] for
    #  index in range(0, cut_off)]

    n_most_common = 100000
    all_passwords = [pwd for pwd in dataset]
    counter = Counter(all_passwords)
    most_common = counter.most_common(n_most_common)

    return most_common, password_slices_dict


def convert_str_to_tensor(string_to_convert: str):
    size = len(string_to_convert)
    converted_tensor = torch.zeros(size, 1).long()
    for index, char in enumerate(string_to_convert):
        converted_tensor[index][0] = string.printable.index(char)
    return converted_tensor.to(device)
