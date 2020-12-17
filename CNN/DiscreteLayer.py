import numpy as np

class DiscreteLayer:
    def __init__(self):
        pass

    def compute(self, x):
        return np.minimum(1, np.maximum(0, np.rint(x)))
