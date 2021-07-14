import os
from abc import abstractmethod
from typing import Type, List

import gdown
import unidecode


class PasswordDataset:
    @abstractmethod
    def get_download_url(self) -> str:
        pass

    @abstractmethod
    def get_dataset_local_path(self) -> str:
        pass


class ClixSense(PasswordDataset):
    def get_download_url(self) -> str:
        return 'https://drive.google.com/uc?id=1S0-1gdzoP-HecS3L5_zStZhvTt9A4q97'

    def get_dataset_local_path(self) -> str:
        return 'pwd_dataset_manager/datasets/ClixSense.txt'


class WebHost(PasswordDataset):
    def get_download_url(self) -> str:
        return 'https://drive.google.com/uc?id=11tsLveuHo3xaVL2DRh3FfuUPG8LEtzYd'

    def get_dataset_local_path(self) -> str:
        return 'pwd_dataset_manager/datasets/000webhost.txt'


class Mate1(PasswordDataset):
    def get_download_url(self) -> str:
        return 'https://drive.google.com/uc?id=10LtJiV9J-Vuy1I8iSxacH4fsPqZamIeB'

    def get_dataset_local_path(self) -> str:
        return 'pwd_dataset_manager/datasets/Mate1.txt'


class DatasetFactory:
    def get(self, dataset_name: str) -> Type[PasswordDataset]:
        if dataset_name == "ClixSense":
            return ClixSense
        elif dataset_name == "000webhost":
            return WebHost
        elif dataset_name == "Mate1":
            return Mate1
        else:
            raise NotImplementedError(f"Dataset: {dataset_name} not supported")


def get_dataset(dataset_klass: Type[PasswordDataset]) -> List[str]:
    local_dataset = dataset_klass().get_dataset_local_path()
    if not os.path.exists(local_dataset):
        gdown.download(dataset_klass().get_download_url(), quiet=True, output=local_dataset)

    with open(local_dataset, "r") as dataset:
        contents = unidecode.unidecode(dataset.read())
        contents = contents.split('\n')
        contents = [content[:-1].strip() for content in contents]
    return contents
