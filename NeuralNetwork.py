from scipy.special import expit as sigmoid
import numpy as np


class NeuralNetwork:
    def __init__(self, neurons_by_layer, learning_rate=0.01, max_iterations=1000):
        self.weights = self.initialize_weights(neurons_by_layer)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.activation_function = sigmoid

    def initialize_weights(self, neurons_by_layer):
        self.weights = []

        for i in range(len(neurons_by_layer) - 1):
            self.weights.append(np.random.normal(0.0, neurons_by_layer[i] ** -0.01,  (neurons_by_layer[i], neurons_by_layer[i+1])))

        return self.weights

    def train(self, x, y, use_batch=True):
        length = len(y)

        if use_batch:
            for i in range(self.max_iterations):
                for batch_f, batch_t in zip(np.array_split(x, 30), np.array_split(y, 30)):
                    batch_deltas = None
                    for f, t in zip(batch_f, batch_t):
                        outputs = self.feed_forward(f)
                        deltas = self.back_propagation(outputs, f, t)
                        if batch_deltas is None:
                            batch_deltas = deltas
                        else:
                            for d, bd in zip(deltas, batch_deltas):
                                bd += d

                    self.update_weights(batch_deltas, length)

        else:
            for i in range(self.max_iterations):
                for a, b in zip(x, y):
                    outputs = self.feed_forward(a)
                    deltas = self.back_propagation(outputs, a, b)
                    self.update_weights(deltas, length)

    def predict(self, x):
        results = []
        for a in x:
            # print(a.shape)
            # print(self.weights[0].shape)
            # exit()
            a = a[None, ...]
            # print(a.shape)
            output = self.activation_function(np.dot(a, self.weights[0]))

            for layer in self.weights[1:]:
                output = self.activation_function(np.dot(output, layer))

            # print(output)
            # print(output.shape)
            # exit()
            if output[0] > 0.5:
                results.append(1.)
            else:
                results.append(0.)

        # return np.array(results, dtype=np.int)
        return np.array(results)

    def predict_(self, x):
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

    for i in range(1):
        nn = NeuralNetwork([2, 100, 100, 1], learning_rate=0.3, max_iterations=50)
        nn.train(features, target, False)
        preds_mask = nn.predict_(features)
        preds_list = nn.predict(features)

        preds_list = preds_list.reshape(target.shape)
        preds_mask = preds_mask.reshape(target.shape)

        print(preds_list.shape)
        print(preds_mask.shape)

        print('Mascara:', (preds_mask == target).mean() * 100)
        print('Lista:', (preds_list == target).mean() * 100)

        print()
        print('=' * 150)
        print()
