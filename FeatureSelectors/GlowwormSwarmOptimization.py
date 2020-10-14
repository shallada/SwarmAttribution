import math
import numpy as np
import pyswarms as ps
import random
import sys
from scipy.spatial import distance as dist


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

PopSize = 100
NIter = int(15000/PopSize)
UseDiscrete = True
DecayRate = 0.15
FitnessWeight = 0.7
StepSize = 0.1
FeatureWeight = 0.0

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
		self.luciferin = (self.luciferin * (1.0 - DecayRate)) + (FitnessWeight * EvaluateMask(mask, x, y, feature_weight = FeatureWeight))

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
	return max(pop, key=lambda worm: EvaluateMask(worm.location, x, y, feature_weight = FeatureWeight))

def SwarmMask(x, y):

	# Input: array of shape (pop_size, features)
	# Output: fitness values of shape (pop_size)

	pop = CreatePopulation(PopSize, len(x[0]))
	for _ in range(NIter):
		#print('loop = '+str(_))
		UpdateLuciferinLevels(pop, x, y)
		UpdatePositions(pop)
		UpdateSensorRadius(pop)

	loc = BestFit(pop, x, y).location
	if UseDiscrete:
		mask = [0 if _ < 0.5 else 1 for _ in loc]
	else:
		mask = loc
	return mask

x, y = LoadFeatures(all_features_file_name)

FeatureWeight = ones_ratio
mask = SwarmMask(x, y)

accuracy = EvaluateMask(mask, x, y, feature_weight=0)

with open(out_file_name, 'w') as out_file:
	out_file.write(str(accuracy)+","+str(mask)+"\n")
