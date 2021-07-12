from random import randint
from typing import List, Tuple


def get_random_password_chunk(dataset: List[str]) -> Tuple[str, str]:
    total_len = len(dataset)
    random_password = dataset[randint(0, total_len - 1)]

    len_password = len(random_password)
    random_cut = randint(1, len_password - 1 - 1)

    inp = random_password[0:len_password - 1]
    expected = random_password[0:len_password]
    return inp, expected
