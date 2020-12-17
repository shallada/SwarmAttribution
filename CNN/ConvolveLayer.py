import numpy as np

class ConvolveLayer:
    def __init__(self, n_filters, filter_size):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.filters = np.random.random_sample((n_filters, filter_size))

    def compute(self, x):
    	maps = np.empty((self.n_filters, x.shape[0] - self.filter_size + 1))
    	for n in range(self.n_filters):
            maps[n] = np.convolve(x, self.filters[n], mode='valid')
    	return maps
