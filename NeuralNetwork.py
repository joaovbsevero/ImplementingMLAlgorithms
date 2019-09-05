import math
import numpy as np


class NeuralNetwork:
    def __init__(self, neurons_by_layer):
        self.weights = self.initialize_weights(neurons_by_layer)

    def initialize_weights(self, neurons_by_layer):
        self.weights = []

        for i in range(len(neurons_by_layer) - 1):
            self.weights.append(np.random.normal(size=(neurons_by_layer[i], neurons_by_layer[i+1])))

        return self.weights

    def feed_forward(self, x):
        previous_output = x.dot(self.weights[0])

        for layer in self.weights[1:]:
            previous_output = previous_output.dot(layer)

        return previous_output


if __name__ == '__main__':
    n = NeuralNetwork([3, 4, 2])
    print(n.feed_forward(np.array([[10, 10, 10]])))


"""
−1/m ∑i=1m ∑k=1Kyk(i)log⁡((hΘ(x(i)))k)+(1−yk(i))log⁡(1−(hΘ(x(i)))k)

[
    input layer (1)
    [
        [W11 W12 W13 W14]
        [W21 W22 W23 W24]
        [W31 W32 W33 W34]
    ]
    hidden layer (2)
    [
        [W11 W12]
        [W21 W22]
        [W31 W32]
        [w41 w42]
    ]
]
"""