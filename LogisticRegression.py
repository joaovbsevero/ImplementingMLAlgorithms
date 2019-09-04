import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid


class LogisticRegression:
    def __init__(self, learning_rate=1e-4, max_iter=100000, threshold=1e-6):
        self.weights = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.threshold = threshold

    def fit(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        self.weights = np.random.normal(size=(x.shape[1] + 1, 1))
        self.weights[0] = 1
        x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        for i in range(self.max_iter):
            old_weights = np.copy(self.weights)
            self.weights -= self.learning_rate * (sigmoid(x.dot(-self.weights)) - y).T.dot(x).T

            if self.check_convergence(old_weights):
                break

        return self

    def predict(self, x):
        if isinstance(x, list):
            x = np.array(x)
            x = np.append(np.ones((x.shape[0], 1)), x)
        return sigmoid(x.dot(-self.weights))

    def check_convergence(self, old_weights):
        for w in (self.weights - old_weights)[0]:
            if abs(w) <= self.threshold:
                print('converged')
                return True


if __name__ == '__main__':
    l = LogisticRegression()

    features = np.array([
        [10, 45, -1],
        [15, 80, 40],
        [20, 120, 10],
        [5, 20, -10],
        [6, 16, -7],
        [3, 31, -2],
        [18, 220, 10],
        [50, 60, 30],
        [11, 51, 400],
        [0, 0, 0]])
    target = np.array([
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [1],
        [1],
        [1],
        [0]])

    attempt = np.array([10, 45, -1])
    attempt = np.append([1], attempt, axis=0)
    print(l.fit(features, target).predict(attempt))
