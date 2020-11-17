import csv
import numpy as np
import random
import sys

from sklearn import svm # preload to solve build problem

from CNNKeras import CNNKeras
from CNNEvaluateMask import EvaluateMask
from LoadFeatures import LoadFeatures
from GeneticLearner import GeneticLearner
from GEFeS import GEFeS

if len(sys.argv) != 5:
	print("Missing parameters")
	print("FORMAT: CNNEvolve.py cnn_type data_set feature_weight run")
	exit()

algorithm = sys.argv[0].split(".")[0]
cnn_type = sys.argv[1]
data_set = sys.argv[2]
feature_weight = float(sys.argv[3])
run_no = int(sys.argv[4])
all_features_file_name = "../Concatenator/"+data_set+"/AllFeatures.txt"
out_file_name = "output/"+algorithm+"-"+cnn_type+"-"+data_set+"-"+str(feature_weight)+"-"+str(run_no)
state_file_name = "state/"+algorithm+"-"+cnn_type+"-"+data_set+"-"+str(feature_weight)+"-"+str(run_no)


labels =    np.array(["N_FILTERS", "KERNEL_SIZE", "POOL_SIZE", "DENSE_SIZE"])
min_vals =  np.array([    2,            1,             2,           2])
max_vals =  np.array([   64,           64,            64,         300])


def wieghts_to_values(weights):
	return np.round(min_vals + ((max_vals - min_vals) * weights)).astype('int')

class CNNInstance:

	def __init__(self):
		self.weights = np.array([])

	def set_weights(self, weights):
		self.weights = weights

	def score(self, x,  y, prev_mask=None):
		gefes = GEFeS(self, prev_mask)
		feature_fitness, feature_mask = gefes.EvolveMasks(x, y)
		cnn_fitness = self.eval_mask(feature_mask, x, y)
		return cnn_fitness, feature_mask

	def eval_mask(self, mask, x, y):
		cnn_vals = wieghts_to_values(self.weights)
		n_outputs = np.amax(y) - np.amin(y) + 1
		try: # catch bad configurations and respond as though they have bad performance
			cnn = CNNKeras(model_name=cnn_type, n_features=x.shape[1], n_outputs=n_outputs, n_filters=cnn_vals[0].item(), kernel_size=cnn_vals[1].item(), pool_size=cnn_vals[2].item(), dense_size=cnn_vals[3].item())
		except:
			return -1
		return EvaluateMask(mask, x, y, cnn, feature_weight=feature_weight)



x, y = LoadFeatures(all_features_file_name)

learner = GeneticLearner(labels.size, len(x[0]), state_file_name)
fitness, cnn_weights, feature_mask = learner.fit(x, y, CNNInstance())
cnn_vals = wieghts_to_values(cnn_weights)

with open(out_file_name, 'w') as out_file:
	out_file.write(str(fitness)+","+str(cnn_vals)+","+str(feature_mask)+"\n")
