from typing import List, Tuple
import numpy as np

from mnist_data.mnist_loader import MNISTDataLoader


class Hyperparameters:
    SIZE: List[int] = [28 * 28, 100, 100, 10]
    LEARNING_RATE: float = 0.1
    EPOCHS: int = 50
    MINI_BATCH_SIZE: int = 100
    # Add lambda hyperparameter
    LMDA: int = 5

    def __str__(self) -> str:
        str_rep = ""
        str_rep += "Hyperparameters set are as follows"
        for hyper_param in self.__annotations__:
            str_rep += f' \n {hyper_param}: {getattr(self, hyper_param)}'
        return str_rep


class NetworkUtils:
    # Replace sigmoid with relu for hidden and softmax for output layer
    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0.0)

    @staticmethod
    def relu_prime(z: np.ndarray) -> np.ndarray:
        return (z > 0.0) * 1

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z)
        return exp_z / sum(exp_z)


class Network:
    def __init__(self, training_data: List[Tuple[np.ndarray, np.ndarray]],
                 testing_data: List[Tuple[np.ndarray, int]],
                 size: List[int], learning_rate: float, epochs: int,
                 mini_batch_size: int, lmda: int):
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
        self.lmda = lmda

    def _init_biases(self):
        for i in range(1, self.num_layers):
            self.biases.append(np.random.randn(self.size[i], 1))

    def _init_weights(self):
        bias_matrix_sizes = [(self.size[x + 1], self.size[x]) for x in range(self.num_layers - 1)]
        # Init weights by dividing by sqrt of each neuron's input size
        for x, y in bias_matrix_sizes:
            std_dev = 1 / np.sqrt(y)
            self.weights.append(np.random.randn(x, y) * std_dev)

    def train(self):
        for epoch in range(self.epochs):
            np.random.shuffle(self.training_data)
            print(f"Start training for epoch: {epoch + 1} of {self.epochs}")

            num_mini_batches = len(self.training_data) // self.mini_batch_size
            mini_batches = self._create_mini_batches()

            for batch, mini_batch in enumerate(mini_batches, start=1):
                self._update_b_w(mini_batch)
                if batch % 300 == 0:
                    self._calc_accuracy(epoch + 1, batch, num_mini_batches)

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
        # Add L2 normalization
        self.weights = [
            np.dot(w, 1 - (self.learning_rate * self.lmda) / len(self.training_data)) -
            ((self.learning_rate / self.mini_batch_size) * nw) for w, nw in zip(self.weights, nabla_wt)
        ]

    def _get_nabla_bias_zeroes(self) -> List[np.ndarray]:
        return [np.zeros(np.shape(bias)) for bias in self.biases]

    def _get_nabla_wt_zeroes(self) -> List[np.ndarray]:
        return [np.zeros(np.shape(wt)) for wt in self.weights]

    def _run_back_propagation(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        nabla_bias = self._get_nabla_bias_zeroes()
        nabla_wt = self._get_nabla_wt_zeroes()

        activations, z_list = self.feedforward(x)
        # Delta for cross entropy
        error_l = self._delta_cross_entropy(activations[-1], y)

        nabla_bias[-1] = error_l
        nabla_wt[-1] = np.dot(error_l, np.transpose(activations[-2]))

        for layer in range(self.num_layers - 2, 0, -1):
            error_l = np.multiply(
                np.dot(np.transpose(self.weights[layer]), error_l), NetworkUtils.relu_prime(z_list[layer - 1])
            )

            nabla_bias[layer - 1] = error_l
            nabla_wt[layer - 1] = np.dot(error_l, activations[layer - 1].transpose())

        return nabla_bias, nabla_wt

    def _delta_cross_entropy(self, a_l: np.ndarray, y: np.ndarray) -> np.ndarray:
        return a_l - y

    def _calc_accuracy(self, epoch: int, batch: int, total_batches: int):
        correct_results = 0
        total_results = len(self.testing_data)
        for x, y in self.testing_data:
            activations, _ = self.feedforward(x)
            logit = activations[-1]
            if np.argmax(logit) == y:
                correct_results += 1
        print(
            f"Accuracy on testing data for epoch {epoch} mini_batch {batch} of {total_batches}: {round((correct_results / total_results) * 100, 2)}"
        )

    def feedforward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        a = x
        activations, z_list = list(), list()
        activations.append(x)
        self._set_relu_activations(a, z_list, activations)
        self._set_softmax_activation(activations[-1], z_list, activations)
        return activations, z_list

    def _set_relu_activations(self, a: np.ndarray, z_list: List[np.ndarray], activations: List[np.ndarray]):
        for layer in range(self.num_layers - 2):
            # hidden layers(relu activation)
            z = np.dot(self.weights[layer], a) + self.biases[layer]
            z_list.append(z)
            a = NetworkUtils.relu(z)
            activations.append(a)

    def _set_softmax_activation(self, a: np.ndarray, z_list: List[np.ndarray], activations: List[np.ndarray]):
        # output later(softmax activation)
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        z_list.append(z)
        a = NetworkUtils.softmax(z)
        activations.append(a)


def train_and_eval():
    training, testing = MNISTDataLoader().load_data_wrapper()
    params = Hyperparameters()
    print(params)
    mlp = Network(training, testing, params.SIZE, params.LEARNING_RATE, params.EPOCHS, params.MINI_BATCH_SIZE,
                  params.LMDA)
    mlp.train()


if __name__ == '__main__':
    train_and_eval()
