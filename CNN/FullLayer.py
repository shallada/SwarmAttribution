import numpy as np

class FullLayer:
    def __init__(self, size):
        self.weights = np.random.random_sample((size, size))
        self.bias = np.random.random_sample((size))

    def compute(self, x):
        return np.dot(x, self.weights) + self.bias
