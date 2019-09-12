import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, input_nodes ** -0.01,
                                                        (input_nodes, hidden_nodes))

        self.weights_hidden_to_hidden = np.random.normal(0.0, hidden_nodes ** -0.01,
                                                         (hidden_nodes, hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, hidden_nodes ** -0.01,
                                                         (hidden_nodes, output_nodes))

        self.lr = learning_rate

        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        for i in range(500):
            n_records = features.shape[0]
            for X, y in zip(features, targets):
                inputs_hidden, hidden_hidden, hidden_outputs = self.forward_pass_train(X)

                delta_weights_i_h, delta_weights_h_h, delta_weights_h_o = self.backpropagation(inputs_hidden, hidden_hidden, hidden_outputs, X, y)

                self.update_weights(delta_weights_i_h, delta_weights_h_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        inputs_hidden = self.activation_function(np.dot(X, self.weights_input_to_hidden))
        hidden_hidden = self.activation_function(np.dot(inputs_hidden, self.weights_hidden_to_hidden))
        hidden_outputs = self.activation_function(np.dot(hidden_hidden, self.weights_hidden_to_output))

        return inputs_hidden, hidden_hidden, hidden_outputs

    def backpropagation(self, inputs_hidden, hidden_hidden, hidden_outputs, X, y):
        output_error_term = hidden_outputs - y
        hidden_outputs_error_term = np.dot(output_error_term, self.weights_hidden_to_output.T) * hidden_hidden * (1 - hidden_hidden)
        hidden_hidden_error_term = np.dot(hidden_outputs_error_term, self.weights_hidden_to_hidden.T) * inputs_hidden * (1 - inputs_hidden)

        delta_weights_i_h = hidden_hidden_error_term * X[:, None]
        delta_weights_h_h = hidden_outputs_error_term * hidden_hidden
        delta_weights_h_o = output_error_term * hidden_outputs

        return delta_weights_i_h, delta_weights_h_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_h, delta_weights_h_o, n_records):
        self.weights_input_to_hidden -= self.lr * delta_weights_i_h / n_records
        self.weights_hidden_to_hidden -= self.lr * delta_weights_h_h / n_records
        self.weights_hidden_to_output -= self.lr * delta_weights_h_o / n_records

    def run(self, features):
        _, _, final_outputs = self.forward_pass_train(features)
        results = (final_outputs > 0.5) * 1.

        return results


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

    for i in range(10):
        nn = NeuralNetwork(2, 20, 1, 0.1)
        nn.train(features, target)
        preds = nn.run(features)

        print((preds == target).mean() * 100)
