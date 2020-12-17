import numpy as np

def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def relu(Z):
	return np.maximum(0,Z)

def fully(x, weights):
	return np.dot(weights, x)

def convolve(x, filters):
	n_filters = filters.shape[0]
	maps = np.empty((n_filters, x.shape[0] - filters.shape[1] + 1))
	for n in range(n_filters):
		maps[n] = np.convolve(x, filters[n], mode='valid')
	return maps

def pool(x, size, stride):
	strides = int((x.shape[1] - size + stride - 1) / stride) + int(((x.shape[1] - size) % stride + size) / size)
	pool = np.empty((x.shape[0], strides))
	for r in range(x.shape[0]):
		for c in range(strides):
			start = c*stride
			end = min(c*stride+size, x.shape[1])
			pool[r, c] = np.max(x[r, start:end])
	return pool


class Network:

	def __init__(self, x_size, n_filters, filter_size, stride, y_size):
		self.x_size = x_size
		self.n_filters = n_filters
		self.filter_size = filter_size
		self.stride = stride
		self.y_size = y_size

		self.in_full_weights = np.random.random_sample((x_size, x_size))
		self.filters = np.random.random_sample((n_filters, filter_size))
		self.out_full_weights = np.random.random_sample((, ))
