import csv
import numpy as np
import random
import sys


from EvaluateMask import EvaluateMask
from LoadFeatures import LoadFeatures


if len(sys.argv) != 2:
	print("include the dataset subfolder name as a parameter")
	exit()

#
# Parameters
#
all_features_file_name = "../Concatenator/"+sys.argv[1]+"/AllFeatures.txt"
NGen = 10000
PopSize = 10
MutationRate = .3
NParents = 2
NSplits = 4
UseDiscrete = False
MaskMin = 0.0
MaskMax = 1.0



def EvaluatePopulation(population, x, y):
	for i in range(len(population)):
		population[i][1] = EvaluateMask(population[i][0], x, y)
		

def CreatePopulation(pop_size, mask_size):
	population = []
	for _ in range(pop_size):
		if UseDiscrete:
			mask = np.random.choice([0, 1], size=(mask_size,)).tolist()
		else:
			mask = [random.random() for j in range(mask_size)]
		accuracy = -1
		population.append([mask, accuracy])
	return population

def ChooseParent(population):
	return random.choice(population)

def ChooseParents(population, n_parents):
	parents = []
	for _ in range(n_parents):
		parent = ChooseParent(population)
		parents.append(parent)
	return parents

def UniformCrossover(parents, mutation_rate):
	mask = []
	for i in range(len(parents[0][0])):
		if random.random() < mutation_rate:
			if UseDiscrete:
				mask.append(random.choice([0, 1]))
			else:
				mask.append(random.random()*(MaskMax-MaskMin)+MaskMin)
		else:
			mask.append(random.choice(parents)[0][i])
	return mask

def CalcMean(population):
	sum = 0.0
	for individual in population:
		sum += individual[1]
	return sum/len(population)

def EvolveMasks(x, y, pop_size=PopSize, n_gen=NGen, mutation_rate=MutationRate, n_parents=NParents):

	population = CreatePopulation(pop_size, len(x[0]))
	EvaluatePopulation(population, x, y)
	population.sort(key = lambda x: x[1], reverse=True)

	for _ in range(n_gen):
		parents = ChooseParents(population, n_parents)
		child_mask = UniformCrossover(parents, mutation_rate)

		accuracy = EvaluateMask(child_mask, x, y)
		#print('child acc = '+str(accuracy))
		# if the new child is better than the worst of the population,
		if accuracy > population[-1][1]:
			child = [child_mask, accuracy]
			population[-1] = child
			population.sort(key = lambda x: x[1], reverse=True)

		best = population[0][1]
		mean = CalcMean(population)
		print("best = "+str(best)+", mean = "+str(mean)+", worst = "+str(population[-1][1]))
	return fitness, population[0][0]

x, y = LoadFeatures(all_features_file_name)
fitness, mask = EvolveMasks(x, y)

accuracy = EvaluateMask(mask, x, y, feature_weight=0)
print("accuracy = "+str(accuracy))


