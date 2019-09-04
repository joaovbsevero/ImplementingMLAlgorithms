import numpy as np
import pandas as pd
from sklearn import datasets


class MultiVariableLinearRegression:
    def __init__(self, learning_rate=0.00001, max_iterations=1000000, threshold=0.0000000001):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.weights = None

    def fit(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        self.weights = np.random.normal(size=(x.shape[0], 1), scale=0.1)
        self.weights[0] = 1

        m = x.shape[1]
        x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        for i in range(self.max_iterations):
            old_weights = np.copy(self.weights)
            self.weights -= (self.predict(x) - y).T.dot(x).T * self.learning_rate/m

            if self.check_convergence(old_weights):
                break

        return self

    def predict(self, x):
        if isinstance(x, list):
            x = np.array(x)
            x = np.append(np.ones((x.shape[0], 1)), x)
        return x.dot(self.weights)

    def check_convergence(self, old_weights):
        for w in (self.weights - old_weights)[0]:
            if abs(w) <= self.threshold:
                print('converged')
                return True


if __name__ == '__main__':
    m = MultiVariableLinearRegression(learning_rate=0.003, max_iterations=1000000, threshold=0.0000000000000001)

    boston = datasets.load_boston()

    features = np.array(boston['data'])
    features = (features - features.mean(axis=1, keepdims=True))/(features.max(axis=1, keepdims=True) - features.min(axis=1, keepdims=True))
    # features = (features - features.mean(axis=1, keepdims=True)) / features.std(axis=1, keepdims=True)
    print(features.mean(), features.min(), features.max())

    target = np.array(boston['target']).reshape(506, 1)

    m.fit(features, target)
    attempt = features[10]
    attempt = np.append([1], attempt, axis=0)
    print(f'Prediction: {m.predict(attempt)[0]} => {target[10][0]}')

