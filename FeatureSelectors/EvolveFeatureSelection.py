import csv
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

#
# Parameters
#
NGen = 14900
PopSize = 100
MutationRate = .02
NParents = 2
NSplits = 4
UseDiscrete = True
MaskMin = 0.0
MaskMax = 1.0
FeatureWeight = 0.0



def EvaluatePopulation(population, x, y):
	for i in range(len(population)):
		population[i][1] = EvaluateMask(population[i][0], x, y, feature_weight = FeatureWeight)
		

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

def ChooseParent(population, sample_size = 2):
	parent_set = random.sample(population, sample_size)
	parent_set.sort(key = lambda x: x[1], reverse=True)
	return parent_set[0]

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

	for n in range(n_gen):
		parents = ChooseParents(population, n_parents)
		child_mask = UniformCrossover(parents, mutation_rate)

		accuracy = EvaluateMask(child_mask, x, y, feature_weight = FeatureWeight)
		#print('child acc = '+str(accuracy))
		# if the new child is better than the worst of the population,
		if accuracy > population[-1][1]:
			child = [child_mask, accuracy]
			population[-1] = child
			population.sort(key = lambda x: x[1], reverse=True)

		best = population[0][1]
		mean = CalcMean(population)
		#print(str(n)+": best = "+str(best)+", mean = "+str(mean)+", worst = "+str(population[-1][1]))
	return best, population[0][0]

x, y = LoadFeatures(all_features_file_name)

FeatureWeight = ones_ratio
fitness, mask = EvolveMasks(x, y)

accuracy = EvaluateMask(mask, x, y, feature_weight=0)

with open(out_file_name, 'w') as out_file:
	out_file.write(str(accuracy)+","+str(mask)+"\n")
