import math
import numpy as np
import random
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

PopSize = 100
NGen = int(15000/PopSize-1)
UseDiscrete = True
FeatureWeight = 0.0
#SocValue = 1.3
#CogValue = 2.8
SocValue = 2.8
CogValue = 1.3

def convertMask(mask):
	if UseDiscrete:
		ret_mask = np.array([0 if _ < 0.5 else 1 for _ in mask])
	else:
		ret_mask = mask
	return ret_mask

class Particle(object):
	glob_best = None
	glob_fit = 0.0

	@classmethod
	def getBest(cls):
		return convertMask(cls.glob_best)

	def __init__(self, mask_size, x, y):
		self.position = np.array([random.random() for j in range(mask_size)])
		self.velocity = np.array([(2.0 * random.random() - 1.0) for j in range(mask_size)])
		self.pbest = self.position
		self.best_fit = EvaluateMask(convertMask(self.position), x, y, feature_weight = FeatureWeight)
		if self.best_fit > Particle.glob_fit:
			Particle.glob_fit = self.best_fit
			Particle.glob_best = self.pbest

	def update(self, x, y):
		phi = SocValue + CogValue
		K = 2/abs(2.0 - phi - math.sqrt(phi**2 - 4.0 * phi))
		socScale = SocValue * random.random()
		cogScale = CogValue * random.random()
		socDelta = self.pbest - self.position
		cogDelta = Particle.glob_best - self.position
		self.velocity = K * (self.velocity + (socScale * socDelta) + (cogScale * cogDelta))
		self.position = self.position + self.velocity
		fit = EvaluateMask(convertMask(self.position), x, y, feature_weight = FeatureWeight)
		if fit > self.best_fit:
			self.pbest = self.position
			self.best_fit = fit
			if self.best_fit > Particle.glob_fit:
				Particle.glob_fit = self.best_fit
				Particle.glob_best = self.pbest

def CreatePopulation(pop_size, mask_size, x, y):
	population = []
	for _ in range(pop_size):
		population.append(Particle(mask_size, x, y))
	return population


def SwarmMask(x, y):

	# Input: array of shape (pop_size, features)
	# Output: accuracy and best mask

	pop = CreatePopulation(PopSize, len(x[0]), x, y)
	for _ in range(NGen):
		for particle in pop:
			particle.update(x, y)
	best_mask = Particle.getBest()
	accuracy = EvaluateMask(best_mask, x, y, feature_weight = 0.0)
	return accuracy, best_mask

x, y = LoadFeatures(all_features_file_name)

FeatureWeight = ones_ratio
accuracy, mask = SwarmMask(x, y)

with open(out_file_name, 'w') as out_file:
	out_file.write(str(accuracy)+","+str(mask.tolist())+"\n")
