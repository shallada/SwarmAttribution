import numpy as np
import pyswarms as ps
import sys


from EvaluateMask import EvaluateMask
from LoadFeatures import LoadFeatures

if len(sys.argv) != 4:
	print("Missing parameters")
	print("FORMAT: algorithm data_set ones_ratio run")
	exit()

algorithm = sys.argv[0].split(".")[0]
data_set = sys.argv[1]
ones_ratio = float(sys.argv[2])
run_no = int(sys.argv[3])
all_features_file_name = "../Concatenator/"+data_set+"/AllFeatures.txt"
out_file_name = "output/"+algorithm+"-"+data_set+"-"+str(ones_ratio)+"-"+str(run_no)

NGen = 150
PopSize = 10
MutationRate = .1
NParents = 2
NSplits = 4
UseDiscrete = True
NeighborRatio = .3
FeatureWeight = 0.0



def EvaluatePopulation(masks, x, y):
	fit = []
	for i in range(len(masks)):
		fit.append(EvaluateMask(masks[i], x, y, feature_weight = FeatureWeight))
	return fit


def SwarmMask(x, y):

	# Input: array of shape (pop_size, features)
	# Output: fitness values of shape (pop_size)
	def fitness(masks):
		return EvaluatePopulation(masks, x, y)

	"""
	if UseDiscrete:
		neighbors = int(PopSize*NeighborRatio)
		options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':neighbors, 'p':2}
		optimizer = ps.discrete.binary.BinaryPSO(n_particles=PopSize, dimensions=len(x[0]), options=options)

	else:
		options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
		optimizer = ps.single.GlobalBestPSO(n_particles=PopSize, dimensions=len(x[0]), options=options)
	"""

	options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
	optimizer = ps.single.GlobalBestPSO(n_particles=PopSize, dimensions=len(x[0]), options=options)

	cost, pos = optimizer.optimize(fitness, iters=NGen)
	if UseDiscrete:
		pos = [0 if _ < 0.5 else 1 for _ in pos]
	return list(pos)


x, y = LoadFeatures(all_features_file_name)

FeatureWeight = ones_ratio
mask = SwarmMask(x, y)

accuracy = EvaluateMask(mask, x, y, feature_weight=0)

with open(out_file_name, 'w') as out_file:
	out_file.write(str(accuracy)+","+str(mask)+"\n")
