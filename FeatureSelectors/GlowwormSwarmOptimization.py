import csv
import math
import numpy as np
import pyswarms as ps
import random
import re
import sys
from scipy.spatial import distance as dist
from sklearn import preprocessing

from EvaluateMask import EvaluateMask

if len(sys.argv) != 2:
	print("include the dataset subfolder name as a parameter")
	exit()

all_features_file_name = "../Concatenator/"+sys.argv[1]+"/AllFeatures.txt"
NIter = 100
PopSize = 10
UseDiscrete = False
DecayRate = 0.1
FitnessWeight = 0.7
StepSize = 2.0


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

class Worm:
	def __init__(self, n_dim):
		self.location = np.random.uniform(low=0, high=1, size=n_dim)
		self.luciferin = 0.5
		self.radius = math.sqrt(n_dim)


	def update_luciferin(self, x, y):
		if UseDiscrete:
			mask = [0 if _ < 0.5 else 1 for _ in self.location]
		else:
			mask = self.location
		self.luciferin = (self.luciferin * (1.0 - DecayRate)) + (FitnessWeight * EvaluateMask(mask, x, y))

	def update_position(self, other):
		delta = np.array(other.location) - np.array(self.location)
		move = StepSize * delta/math.sqrt(np.sum(delta ** 2.0))
		self.location = np.array(self.location) + move

def CreatePopulation(pop_size, n_dim):
	pop = []
	for _ in range(pop_size):
		pop.append(Worm(n_dim))
	return pop

def UpdateLuciferinLevels(pop, x, y):
	for worm in pop:
		worm.update_luciferin(x,y)

def GetNeighbors(worm, pop):
	neighbors = []
	for w in pop:
		if (dist.euclidean(worm.location, w.location) < worm.radius) and (worm.luciferin < w.luciferin):
			neighbors.append(w)
	return neighbors

def SelectNeighbor(worm, neighbors):
	deltas = []
	sum = 0.0
	for w in neighbors:
		delta = w.luciferin - worm.luciferin
		deltas.append(delta)
		sum += delta
	weights = np.array(deltas)/sum
	return random.choices(neighbors, weights)[0]
		

def UpdatePositions(pop):
	for worm in pop:
		neighbors = GetNeighbors(worm, pop)
		if len(neighbors) > 0:
			neighbor = SelectNeighbor(worm, neighbors)
			worm.update_position(neighbor)

def UpdateSensorRadius(pop):
	pass

def BestFit(pop, x, y):
	return max(pop, key=lambda worm: EvaluateMask(worm.location, x, y))

def SwarmMask(x, y):

	# Input: array of shape (pop_size, features)
	# Output: fitness values of shape (pop_size)

	pop = CreatePopulation(PopSize, len(x[0]))
	for _ in range(NIter):
		print('loop = '+str(_))
		UpdateLuciferinLevels(pop, x, y)
		UpdatePositions(pop)
		UpdateSensorRadius(pop)

	return BestFit(pop, x, y).location

x, y = LoadFeatures(all_features_file_name)
mask = SwarmMask(x, y)

if UseDiscrete:
	mask = [0 if _ < 0.5 else 1 for _ in mask]

print(mask)

accuracy = EvaluateMask(mask, x, y, feature_weight=0)
print("accuracy = "+str(accuracy))