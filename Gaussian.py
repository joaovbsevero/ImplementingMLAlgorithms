import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Gaussian:
    def __init__(self):
        self.means = np.array([])
        self.variances = np.array([])

    def train(self, x):
        df = pd.DataFrame(x)
        for column in df.columns:
            self.means = np.append(self.means, df[column].mean())
            self.variances = np.append(self.variances, df[column].var(ddof=0))

    def predict(self, x):
        results = []
        for instance in x:
            product = np.prod((1/np.sqrt(2*np.pi*np.sqrt(self.variances))) * np.exp(-(((instance - self.means) ** 2)/(2 * self.variances))))
            results.append(1 if product < 0.03 else 0)

        return results


if __name__ == '__main__':
    instances = np.random.normal(10, 5, (1000, 2))
    g = Gaussian()
    g.train(instances)
    # plt.hist(instances)
    # plt.show()
    print(g.predict([[20, 20], [10, 10]]))

