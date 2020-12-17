import numpy as np

class PoolLayer:
    def __init__(self, filter_size, stride):
        self.filter_size = filter_size
        self.stride = stride

    def compute(self, x):
    	strides = int((x.shape[1] - self.filter_size + self.stride - 1) / self.stride) + int(((x.shape[1] - self.filter_size) % self.stride + self.filter_size) / self.filter_size)
    	pool = np.empty((x.shape[0], strides))
    	for r in range(x.shape[0]):
    		for c in range(strides):
    			start = c*self.stride
    			end = min(c*self.stride+self.filter_size, x.shape[1])
    			pool[r, c] = np.max(x[r, start:end])
    	return pool
