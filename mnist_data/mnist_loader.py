import gzip
import shutil
import gdown

from typing import Tuple, List, Union

import numpy as np

import struct


class MNISTDataLoader:
    # explanation of idx file formats: http://yann.lecun.com/exdb/mnist/
    # help wrt parsing data: https://stackoverflow.com/a/53181925

    TRAINING_DATA_URL: str = 'https://drive.google.com/uc?id=1pmI9wAdNtJkOvkJpdTqM9bmIAwPkGyMU'
    TRAINING_DATA_LABELS_URL: str = 'https://drive.google.com/uc?id=1R8BZL67U1N0GUGnf6AQIBZNVDCWO9QLS'
    TESTING_DATA_URL: str = 'https://drive.google.com/uc?id=10FdcUHw3BcQAU6keKaUwtDwJm4sC00Hu'
    TESTING_DATA_LABELS_URL: str = 'https://drive.google.com/uc?id=1GvsacEnI1eQ1vYZM-oYdERvaE2SPh0Lj'

    def load_data_wrapper(self):
        testing_data_tuple = self.load_data_as_ndarray(self.TESTING_DATA_URL, self.TESTING_DATA_LABELS_URL, False)
        training_data_tuple = self.load_data_as_ndarray(self.TRAINING_DATA_URL, self.TRAINING_DATA_LABELS_URL, True)
        return training_data_tuple, testing_data_tuple

    def load_data_wrapper_with_validation(self):
        testing_data_tuple = self.load_data_as_ndarray(self.TESTING_DATA_URL, self.TESTING_DATA_LABELS_URL, False)
        training_data_tuple = self.load_data_as_ndarray(self.TRAINING_DATA_URL, self.TRAINING_DATA_LABELS_URL, True)

        slice_threshold = len(training_data_tuple) - len(testing_data_tuple)
        training_data_tuple, validation_data_tuple = training_data_tuple[:slice_threshold], training_data_tuple[
                                                                                            slice_threshold:]
        self._convert_label_to_int(validation_data_tuple)
        return training_data_tuple, testing_data_tuple, validation_data_tuple

    def load_data_as_ndarray(self, data_file_url: str, data_labels_file_url: str, train: bool) -> List[
        Tuple[np.ndarray, Union[int, np.ndarray]]]:
        uncompressed_dataset = self._download_and_uncompressed_file(data_file_url)
        uncompressed_labels = self._download_and_uncompressed_file(data_labels_file_url)
        pixel_data = self._get_pixel_data(uncompressed_dataset)
        label_data = self._get_labels(uncompressed_labels)
        zipped_data = [
            (x.reshape(784, 1), self._one_hot_enc(y) if train else y[0]) for x, y in zip(pixel_data, label_data)
        ]
        return zipped_data

    def _one_hot_enc(self, y: np.ndarray):
        one_hot_vector = np.zeros((10, 1))
        one_hot_vector[y[0]][0] = 1
        return one_hot_vector

    def _download_and_uncompressed_file(self, url: str) -> str:
        downloaded_gzip = gdown.download(url)
        decompressed_data_file = self._write_decompressed_data(downloaded_gzip)
        return decompressed_data_file

    def _write_decompressed_data(self, downloaded_gzip: str) -> str:
        with gzip.open(downloaded_gzip, 'rb') as compressed:
            uncompressed_dataset = downloaded_gzip.replace('.gz', '')
            with open(uncompressed_dataset, 'wb') as decompressed:
                shutil.copyfileobj(compressed, decompressed)
        return uncompressed_dataset

    def _get_pixel_data(self, data_file: str) -> np.ndarray:
        with open(data_file, "rb") as dataset:
            _, num_data = struct.unpack(">II", dataset.read(8))
            num_rows, num_colums = struct.unpack(">II", dataset.read(8))
            pixel_data = np.fromfile(dataset, dtype=np.uint8) / 255
            pixel_data = pixel_data.reshape((num_data, num_rows * num_colums))
        return pixel_data

    def _get_labels(self, data_labels_file: str) -> Union[np.ndarray, int]:
        with open(data_labels_file, "rb") as labels:
            _, num_data = struct.unpack(">II", labels.read(8))
            label_data = np.fromfile(labels, dtype=np.uint8)
            label_data = label_data.reshape((num_data, -1))
        return label_data

    def _convert_label_to_int(self, validation_data_tuple: List[Tuple[np.ndarray, np.ndarray]]):
        for index, (inp, label) in enumerate(validation_data_tuple):
            validation_data_tuple[index] = (inp, label.argmax())


if __name__ == '__main__':
    MNISTDataLoader().load_data_wrapper()
