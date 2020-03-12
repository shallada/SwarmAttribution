import csv
import numpy as np
import pyswarms as ps
import re
import sys
from sklearn import preprocessing

from EvaluateMask import EvaluateMask


if len(sys.argv) != 2:
	print("include the dataset subfolder name as a parameter")
	exit()



all_features_file_name = "../Concatenator/"+sys.argv[1]+"/AllFeatures.txt"
NGen = 1000
PopSize = 10
MutationRate = .1
NParents = 2
NSplits = 4
UseDiscrete = False
NeighborRatio = .3

def LoadFeatures(file_name):

	X = []
	Y = []
	with open(file_name) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			y = re.split("_", row[0])[0]
			Y.append(y)
			x = [float(f) for f in row[1:]]
			X.append(x)
	Y = preprocessing.LabelEncoder().fit_transform(Y)
	return np.array(X), np.array(Y)



def EvaluatePopulation(masks, x, y):
	fit = []
	for i in range(len(masks)):
		fit.append(EvaluateMask(masks[i], x, y))
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
mask = SwarmMask(x, y)

accuracy = EvaluateMask(mask, x, y, feature_weight=0)
print("accuracy = "+str(accuracy))


