from typing import List, Tuple
import numpy as np

from mnist_data.mnist_loader import MNISTDataLoader


class Hyperparameters:
    SIZE: List[int] = [28 * 28, 30, 10]
    LEARNING_RATE: float = 3
    EPOCHS: int = 10
    MINI_BATCH_SIZE: int = 10


class NetworkUtils:
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z: np.ndarray) -> np.ndarray:
        return NetworkUtils.sigmoid(z) * (1 - NetworkUtils.sigmoid(z))


class Network:
    def __init__(self, training_data: List[Tuple[np.ndarray, np.ndarray]],
                 testing_data: List[Tuple[np.ndarray, int]],
                 size: List[int], learning_rate: float, epochs: int,
                 mini_batch_size: int):
        self.training_data = training_data
        self.testing_data = testing_data
        self.size = size
        self.num_layers = len(size)
        self.learning_rate = learning_rate
        self.biases = []
        self.weights = []
        self._init_biases()
        self._init_weights()
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

    def _init_biases(self):
        for i in range(1, self.num_layers):
            self.biases.append(np.random.randn(self.size[i], 1))

    def _init_weights(self):
        bias_matrix_sizes = [(self.size[x + 1], self.size[x]) for x in range(self.num_layers - 1)]
        for x, y in bias_matrix_sizes:
            self.weights.append(np.random.randn(x, y))

    def train(self):
        for epoch in range(self.epochs):
            np.random.shuffle(self.training_data)
            print(f"Start training for epoch: {epoch + 1} of {self.epochs}")

            num_mini_batches = len(self.training_data) // self.mini_batch_size
            mini_batches = self._create_mini_batches()

            for batch, mini_batch in enumerate(mini_batches, start=1):
                if batch % 1000 == 0:
                    print(f"calculating SGD for: Batch {batch}/{num_mini_batches} of epoch: {epoch}")
                self._update_b_w(mini_batch)
            self._calc_accuracy()

    def _create_mini_batches(self) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        mini_batches = [
            self.training_data[multiple:multiple + self.mini_batch_size] for multiple in
            range(0, len(self.training_data), self.mini_batch_size)
        ]
        return mini_batches

    def _update_b_w(self, mini_batch: List[Tuple[np.ndarray, np.ndarray]]):
        nabla_bias = self._get_nabla_bias_zeroes()
        nabla_wt = self._get_nabla_wt_zeroes()

        for x, y in mini_batch:
            del_bias, del_wt = self._run_back_propagation(x, y)

            nabla_bias = [curr_b + del_b for curr_b, del_b in zip(nabla_bias, del_bias)]
            nabla_wt = [curr_wt + del_w for curr_wt, del_w in zip(nabla_wt, del_wt)]

        self.biases = [
            b - ((self.learning_rate / self.mini_batch_size) * nb) for b, nb in zip(self.biases, nabla_bias)
        ]
        self.weights = [
            w - ((self.learning_rate / self.mini_batch_size) * nw) for w, nw in zip(self.weights, nabla_wt)
        ]

    def _get_nabla_bias_zeroes(self) -> List[np.ndarray]:
        return [np.zeros(np.shape(bias)) for bias in self.biases]

    def _get_nabla_wt_zeroes(self) -> List[np.ndarray]:
        return [np.zeros(np.shape(wt)) for wt in self.weights]

    def _run_back_propagation(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        nabla_bias = self._get_nabla_bias_zeroes()
        nabla_wt = self._get_nabla_wt_zeroes()

        activations = []
        z_list = []

        a = x
        activations.append(a)

        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            z_list.append(z)

            a = NetworkUtils.sigmoid(z)
            activations.append(a)

        error_l = np.multiply(self._nabla_a(activations[-1], y), NetworkUtils.sigmoid_prime(z_list[-1]))
        nabla_bias[-1] = error_l
        nabla_wt[-1] = np.dot(error_l, np.transpose(activations[-2]))

        for layer in range(self.num_layers - 2, 0, -1):
            error_l = np.multiply(
                np.dot(np.transpose(self.weights[layer]), error_l), NetworkUtils.sigmoid_prime(z_list[layer - 1])
            )

            nabla_bias[layer - 1] = error_l
            nabla_wt[layer - 1] = np.dot(error_l, activations[layer - 1].transpose())

        return nabla_bias, nabla_wt

    def _nabla_a(self, a_l: np.ndarray, y: np.ndarray) -> np.ndarray:
        return a_l - y

    def _calc_accuracy(self):
        correct_results = 0
        total_results = len(self.testing_data)
        for x, y in self.testing_data:
            logit = self.feedforward(x)
            if np.argmax(logit) == y:
                correct_results += 1
        print(f"Accuracy on testing data: {round((correct_results / total_results) * 100, 2)}")

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        a = x
        for layer in range(self.num_layers - 1):
            a = NetworkUtils.sigmoid(np.dot(self.weights[layer], a) + self.biases[layer])
        return a


def train_and_eval():
    training, testing = MNISTDataLoader().load_data_wrapper()
    params = Hyperparameters
    mlp = Network(training, testing, params.SIZE, params.LEARNING_RATE, params.EPOCHS, params.MINI_BATCH_SIZE)
    mlp.train()


if __name__ == '__main__':
    train_and_eval()
