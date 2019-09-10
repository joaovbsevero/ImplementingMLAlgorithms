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
            self.weights.append(np.random.normal(0.0, neurons_by_layer[i] ** -0.0001,  size=(neurons_by_layer[i], neurons_by_layer[i+1])))

        return self.weights

    def train(self, x, y, use_batch=True):
        length = len(y)

        if use_batch:
            for i in range(self.max_iterations):
                for batch_f, batch_t in zip(np.array_split(x, 20), np.array_split(y, 20)):
                    batch_deltas = None
                    for f, t in zip(batch_f, batch_t):
                        outputs = self.feed_forward(f)
                        deltas = self.back_propagation(outputs, f, t)
                        for d in deltas:
                            d = np.clip(d, -200, 200)
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
                    for d in deltas:
                        d = np.clip(d, -200, 200)
                    self.update_weights(deltas, length)

    def predict(self, x):
        results = []
        for a in x:
            output = self.activation_function(np.dot(a, self.weights[0]))

            for layer in self.weights[1:]:
                output = np.dot(output, layer)

            if output[0] > 0:
                results.append(1)
            else:
                results.append(0)

        return results

    def feed_forward(self, x):
        outputs = [self.activation_function(np.dot(x, self.weights[0]))]

        for layer in self.weights[1:]:
            outputs.append(np.dot(outputs[-1], layer))

        return outputs

    def back_propagation(self, outputs, x, y):
        hidden_errors = [outputs[-1] - y]

        for layer, output in zip(reversed(self.weights), reversed(outputs[:-1])):
            hidden_errors.append(np.dot(hidden_errors[-1], layer.T) * output * (1 - output))

        deltas = [hidden_errors[-1] * x[:, None]]
        for error, output in zip(reversed(hidden_errors[:-1]), outputs[1:]):
            deltas.append((error * output))

        return deltas

    def update_weights(self, delta, length):
        for layer, delt in zip(self.weights, delta):
            layer += self.learning_rate * delt / length


def generate_data(target):
    features = []
    for t in target:
        if t == 0:
            features.append([np.random.randint(10, 20), np.random.randint(10, 20)])
        else:
            features.append([np.random.randint(50, 80), np.random.randint(50, 80)])

    return features


if __name__ == '__main__':
    for _ in range(15):
        n = NeuralNetwork([2, 10, 10, 4, 4, 2, 2, 1])
        target = np.array(np.random.choice([0, 1], size=(1000, 1), p=[3./4, 1./4]))
        features = np.array(generate_data(target))

        n.train(features, target, False)
        preds = n.predict(features)
        acc = (preds == target).mean()
        print(acc * 100)
