import csv
import numpy as np
import random
import sys



from CNNKeras import CNNKeras
from CNNEvaluateMask import EvaluateMask
from LoadFeatures import LoadFeatures
from GeneticLearner import GeneticLearner
from GEFeS import GEFeS

if len(sys.argv) != 4:
	print("Missing parameters")
	print("FORMAT: algorithm data_set feature_weight run")
	exit()

algorithm = sys.argv[0].split(".")[0]
data_set = sys.argv[1]
feature_weight = float(sys.argv[2])
run_no = int(sys.argv[3])
all_features_file_name = "../Concatenator/"+data_set+"/AllFeatures.txt"
out_file_name = "output/"+algorithm+"-"+data_set+"-"+str(feature_weight)+"-"+str(run_no)


N_GEN = 100

labels =    np.array(["N_FILTERS", "KERNEL_SIZE", "POOL_SIZE", "DENSE_SIZE"])
min_vals =  np.array([    2,            2,             2,           2])
max_vals =  np.array([   64,           64,            64,         300])




class EvolveCNN:

	def __init__(self):
		self.weights = np.array([])

	def set_weights(self, weights):
		self.weights = weights

	def score(self, x,  y):
		gefes = GEFeS(self)
		fitness, mask = gefes.EvolveMasks(x, y)
		accuracy = self.eval_mask(mask, x, y)
		return accuracy

	def eval_mask(self, mask, x, y):
		cnn_vals = np.round(min_vals + ((max_vals - min_vals) * self.weights)).astype('int')
		n_outputs = np.amax(y) - np.amin(y) + 1
		cnn = CNNKeras(x.shape[1], n_outputs, n_filters=cnn_vals[0].item(), kernel_size=cnn_vals[1].item(), pool_size=cnn_vals[2].item(), dense_size=cnn_vals[3].item())
		return EvaluateMask(mask, x, y, cnn, feature_weight=feature_weight)



x, y = LoadFeatures(all_features_file_name)

learner = GeneticLearner(labels.size)
learner.fit(x, y, EvolveCNN())

with open(out_file_name, 'w') as out_file:
	out_file.write(str(accuracy)+","+str(mask)+"\n")
