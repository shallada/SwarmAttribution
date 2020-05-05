import numpy as np
import pyswarms as ps
import sys

from EvaluateMask import EvaluateMask
from LoadFeatures import LoadFeatures

if len(sys.argv) != 2:
	print("include the dataset subfolder name as a parameter")
	exit()



all_features_file_name = "../Concatenator/"+sys.argv[1]+"/AllFeatures.txt"
NGen = 50
PopSize = 10
MutationRate = .1
NParents = 2
NSplits = 4
UseDiscrete = False
NeighborRatio = .3
NRuns = 30
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


	if UseDiscrete:
		neighbors = int(PopSize*NeighborRatio)
		options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':neighbors, 'p':2}
		optimizer = ps.discrete.binary.BinaryPSO(n_particles=PopSize, dimensions=len(x[0]), options=options)

	else:
		options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
		optimizer = ps.single.GlobalBestPSO(n_particles=PopSize, dimensions=len(x[0]), options=options)

	cost, pos = optimizer.optimize(fitness, iters=NGen)
	return pos


x, y = LoadFeatures(all_features_file_name)

for fw in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
	for n in range(NRuns):
		FeatureWeight = fw
		mask = SwarmMask(x, y)

		accuracy = EvaluateMask(mask, x, y, feature_weight=0)
		print(str(n)+": Feature Weight = "+str(fw)+", accuracy = "+str(accuracy), flush = True)

