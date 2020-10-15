import csv
import numpy as np
import random
import sys



from LoadFeatures import LoadFeatures


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


class GEFeS:

	def __init__(self, evaluator):
		self.evaluator = evaluator

	def EvaluatePopulation(self, population, x, y):
		for i in range(len(population)):
			population[i][1] = self.evaluator.eval_mask(population[i][0], x, y)


	def CreatePopulation(self, pop_size, mask_size):
		population = []
		for _ in range(pop_size):
			if UseDiscrete:
				mask = np.random.choice([0, 1], size=(mask_size,)).tolist()
			else:
				mask = [random.random() for j in range(mask_size)]
			accuracy = -1
			population.append([mask, accuracy])
		return population

	def ChooseParent(self, population, sample_size = 2):
		parent_set = random.sample(population, sample_size)
		parent_set.sort(key = lambda x: x[1], reverse=True)
		return parent_set[0]

	def ChooseParents(self, population, n_parents):
		parents = []
		for _ in range(n_parents):
			parent = self.ChooseParent(population)
			parents.append(parent)
		return parents

	def UniformCrossover(self, parents, mutation_rate):
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

	def CalcMean(self, population):
		sum = 0.0
		for individual in population:
			sum += individual[1]
		return sum/len(population)

	def EvolveMasks(self, x, y, pop_size=PopSize, n_gen=NGen, mutation_rate=MutationRate, n_parents=NParents):

		population = self.CreatePopulation(pop_size, len(x[0]))
		self.EvaluatePopulation(population, x, y)
		population.sort(key = lambda x: x[1], reverse=True)

		for n in range(n_gen):
			parents = self.ChooseParents(population, n_parents)
			child_mask = self.UniformCrossover(parents, mutation_rate)

			accuracy = self.evaluator.eval_mask(child_mask, x, y)
			#print('child acc = '+str(accuracy))
			# if the new child is better than the worst of the population,
			if accuracy > population[-1][1]:
				child = [child_mask, accuracy]
				population[-1] = child
				population.sort(key = lambda x: x[1], reverse=True)

			best = population[0][1]
			mean = self.CalcMean(population)
			print(str(n)+": best = "+str(best)+", mean = "+str(mean)+", worst = "+str(population[-1][1]))
		return best, population[0][0]
