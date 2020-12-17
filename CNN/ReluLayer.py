import numpy as np

class ReluLayer:
    def __init__(self):
        pass

    def compute(self, x):
        return np.maximum(0,x)
