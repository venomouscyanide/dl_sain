import os
from abc import abstractmethod
from typing import Type, List

import gdown


class PasswordDataset:
    @abstractmethod
    def get_download_url(self):
        pass

    @abstractmethod
    def get_dataset_local_path(self):
        pass


class ClixSense(PasswordDataset):
    def get_download_url(self):
        return 'https://drive.google.com/uc?id=1S0-1gdzoP-HecS3L5_zStZhvTt9A4q97'

    def get_dataset_local_path(self):
        return 'pwd_dataset_manager/datasets/ClixSense.txt'


class DatasetFactory:
    def get(self, dataset_name: str):
        if dataset_name == "ClixSense":
            return ClixSense
        else:
            raise NotImplementedError(f"Dataset: {dataset_name} not supported")


def get_dataset(dataset_klass: Type[PasswordDataset]) -> List[str]:
    local_dataset = dataset_klass().get_dataset_local_path()
    if not os.path.exists(local_dataset):
        gdown.download(dataset_klass().get_download_url(), quiet=True, output=local_dataset)

    with open(local_dataset, "r") as dataset:
        contents = dataset.readlines()
        contents = [content[:-1] for content in contents]
    return contents
