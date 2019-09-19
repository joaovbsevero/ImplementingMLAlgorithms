import copy
from scipy.special import expit as sigmoid
import numpy as np


class NeuralNetwork:
    def __init__(self, neurons_by_layer, learning_rate=0.01, max_iterations=1000, batch_size=100, beta=.9):
        self.weights = self.initialize_weights(neurons_by_layer)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.activation_function = sigmoid
        self.batch_size = batch_size
        self.beta = beta

    def initialize_weights(self, neurons_by_layer):
        self.weights = []

        for i in range(len(neurons_by_layer) - 1):
            self.weights.append(np.random.normal(0.0, neurons_by_layer[i] ** -0.01,  (neurons_by_layer[i], neurons_by_layer[i+1])))

        return self.weights

    def train(self, x, y, use_batch=True):

        if use_batch:
            self._train_mini_batch(x, y)
        else:
            self._train_batch(x, y)

    def _train_mini_batch(self, x, y):
        length = len(y)
        for i in range(self.max_iterations):
            momentum = []
            for batch_f, batch_t in zip(np.array_split(x, self.batch_size), np.array_split(y, self.batch_size)):
                batch_deltas = None
                for f, t in zip(batch_f, batch_t):
                    outputs = self.feed_forward(f)
                    deltas = self.back_propagation(outputs, f, t)
                    if not momentum:
                        for d in deltas:
                            momentum.append(np.zeros(d.shape))

                    v_deltas = []
                    for d, m in zip(deltas, momentum):
                        v_deltas.append(self.beta * m + (1 - self.beta) * d)

                    if batch_deltas is None:
                        batch_deltas = deltas
                    else:
                        for d, bd in zip(deltas, batch_deltas):
                            bd += d

                old_weights = copy.deepcopy(self.weights)
                self.update_weights(batch_deltas, length)

                differences = []
                for old_layer, new_layer in zip(old_weights, self.weights):
                    if np.all(abs(old_layer - new_layer) < 1e-8):
                        differences.append(True)
                        print('converged')
                    else:
                        differences.append(False)
                        break

                if all(differences):
                    return

                differences.clear()

    def _train_batch(self, x, y):
        length = len(y)
        for i in range(self.max_iterations):
                momentum = []
                for a, b in zip(x, y):
                    outputs = self.feed_forward(a)
                    deltas = self.back_propagation(outputs, a, b)
                    if not momentum:
                        for d in deltas:
                            momentum.append(np.zeros(d.shape))

                    v_deltas = []
                    for d, m in zip(deltas, momentum):
                        v_deltas.append(self.beta * m + (1 - self.beta) * d)

                    old_weights = copy.deepcopy(self.weights)
                    self.update_weights(v_deltas, length)

                    differences = []
                    for old_layer, new_layer in zip(old_weights, self.weights):
                        if np.all(abs(old_layer - new_layer) < 1e-8):
                            differences.append(True)
                            print('converged')
                        else:
                            differences.append(False)
                            break

                    if all(differences):
                        return

                    differences.clear()

    def predict(self, x):
        return np.array((self.feed_forward(x)[-1] > 0.5) * 1)

    def feed_forward(self, x):
        output = self.activation_function(np.dot(x, self.weights[0]))
        outputs = [output]

        for layer in self.weights[1:]:
            output = self.activation_function(np.dot(output, layer))
            outputs.append(output)

        return outputs

    def back_propagation(self, outputs, x, y):
        error = outputs[-1] - y
        hidden_errors = [error]

        for layer, output in zip(reversed(self.weights), reversed(outputs[:-1])):
            error = np.dot(error, layer.T) * output * (1 - output)
            hidden_errors.append(error)

        deltas = [hidden_errors[-1] * x[:, None]]
        for error, output in zip(reversed(hidden_errors[:-1]), outputs[1:]):
            deltas.append((error * output))

        return deltas

    def update_weights(self, delta, length):
        for layer, delt in zip(self.weights, delta):
            layer -= self.learning_rate * delt / length


def generate_data(target):
    features = []
    for t in target:
        if t == 0:
            features.append([np.random.randint(5, 10), np.random.randint(5, 10)])
        else:
            features.append([np.random.randint(20, 30), np.random.randint(20, 30)])

    return features


if __name__ == '__main__':
    target = np.array(np.random.choice([0, 1], size=(1000, 1), p=[1. / 2, 1. / 2]))
    features = np.array(generate_data(target))

    # a = np.array([[2, 3, 4, 5, 6, 8], [3, 3, 4, 5, 6, 7], [2, 3, 4, 6, 6, 7]])
    # b = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    # print(np.all(abs(a - b) > 1))

    for i in range(10):
        nn = NeuralNetwork([2, 20, 20, 1], learning_rate=0.3, max_iterations=500, batch_size=128)
        nn.train(features, target, True)
        preds_list = nn.predict(features)

        preds_list = preds_list.reshape(target.shape)

        print((preds_list == target).mean() * 100)

        print()
        print('=' * 150)
        print()
