import numpy as np

class SigmoidLayer:
    def __init__(self):
        pass

    def compute(self, x):
        return 1/(1+np.exp(-x))
