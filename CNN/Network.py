import numpy as np
import sys

from EvaluateMask import EvaluateMask
from LoadFeatures import LoadFeatures

from FullLayer import FullLayer
from ConvolveLayer import ConvolveLayer
from ReluLayer import ReluLayer
from PoolLayer import PoolLayer
from DiscreteLayer import DiscreteLayer

class Network:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def compute(self, x):
        z = x
        for layer in self.layers:
            print('shape = ' + str(z.shape))
            z = layer.compute(z)
        return z

def get_mask(x):
    n_features = len(x[0])
    n_filters = 20
    filter_size = 5
    pool_filter_size = 4
    pool_stride = 4

    net = Network()
    net.addLayer(FullLayer(n_features))
    net.addLayer(ConvolveLayer(n_filters, filter_size))
    net.addLayer(ReluLayer())
    net.addLayer(PoolLayer(pool_filter_size, pool_stride))
    net.addLayer(FullLayer(n_features))
    net.addLayer(DiscreteLayer())

    y = net.compute(x)
    return y

if len(sys.argv) != 4:
	print("Missing parameters")
	print("FORMAT: algorith data_set ones_ratio run")
	exit()

algorithm = sys.argv[0].split(".")[0]
data_set = sys.argv[1]
ones_ratio = float(sys.argv[2])
run_no = int(sys.argv[3])
all_features_file_name = "../Concatenator/"+data_set+"/AllFeatures.txt"
out_file_name = "output/"+algorithm+"-"+data_set+"-"+str(ones_ratio)+"-"+str(run_no)

x, y = LoadFeatures(all_features_file_name)


NEvals = 15000


with open(out_file_name, 'w') as out_file:

	best_accuracy = 0.0
	for _ in range(NEvals):
		mask = get_mask(x)
		accuracy = EvaluateMask(mask, x, y, feature_weight=ones_ratio)
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_mask = mask

	accuracy = EvaluateMask(mask, x, y, feature_weight=0)

	out_file.write(str(accuracy)+","+str(best_mask)+"\n")
