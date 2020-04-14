import math
import numpy as np
import random
import sys


from EvaluateMask import EvaluateMask
from LoadFeatures import LoadFeatures

if len(sys.argv) != 2:
	print("include the dataset subfolder name as a parameter")
	exit()

all_features_file_name = "../Concatenator/"+sys.argv[1]+"/AllFeatures.txt"
NIter = 1000
PopSize = 10
UseDiscrete = True
DecayRate = 0.1
NBuckets = 100
BucketRadius = 3
FitScale = 1.0
if UseDiscrete:
	NBuckets = 2
	BucketRadius = 0



class Bucket:
	def __init__(self):
		self.count = 1.0
		self.mean = 0.5
	def __str__(self):
		return "mean: "+str(self.mean)+" count: "+str(self.count)
	def value(self):
		return self.mean
	def add_sample(self, value, count):
		self.mean = ((self.mean * self.count) + (value * count)) / (self.count + count)
		self.count += count

def AntToMask(ant, buckets):
	mask = []
	for i in range(len(buckets)): # for each feature
		if UseDiscrete:
			mask.append(ant[i])
		else:
			mask.append(buckets[i][ant[i]].value())
	return mask

def EvaluatePopulation(ants, buckets, x, y):
	fit = []
	for i in range(len(ants)):
		mask = AntToMask(ants[i], buckets)

		fitness = EvaluateMask(mask, x, y)
		fit.append(fitness)

	return list(fit)

def CreateFeatureBuckets(n_features, n_buckets = NBuckets):
	buckets = []
	for f in range(n_features):
		features = []
		for b in range(n_buckets):
			features.append(Bucket())
		buckets.append(features)
	return buckets

def ChooseWeightedValueIndex(values):
	a = np.array(values)
	if sum(a) > 0.0:
		index = np.random.choice(np.arange(a.size), p=a/sum(a))
	else:
		index = np.random.randint(a.size)
	return index

def BucketListToValueList(bucket_list):
	values = []
	for bucket in bucket_list:
		values.append(bucket.value())
	return values

def GeneratePath(buckets):
	path = []
	for n in range(len(buckets)):
		index = ChooseWeightedValueIndex(BucketListToValueList(buckets[n]))
		path.append(index)
	return path

def CreateAnts(buckets, n_ants=PopSize):
	ants = []
	for _ in range(n_ants):
		ant = GeneratePath(buckets)
		ants.append(ant)
	return ants

def ApplyDecay(buckets, decay_rate = DecayRate):
	pass

def ApplyPheromones(buckets, ants, fit, fit_scale = FitScale, bucket_radius = BucketRadius):

	n_ants = len(ants)
	n_features = len(buckets)
	n_buckets = len(buckets[0])

	for a in range(n_ants): # for each ant
		for f in range(n_features): # for each feature

			hot_index = ants[a][f]
			buckets[f][hot_index].add_sample(fit[a], 1.0)


def BucketsToValues(buckets):
	values = []
	for n in range(len(buckets)):
		values.append(BucketListToValueList(buckets[n]))
	return values

def DeriveMask(buckets):
	values = BucketsToValues(buckets)

	mask = []
	for n in range(len(values)):
		indices = [i for i, x in enumerate(values[n]) if x == max(values[n])]
		index = np.random.choice(indices)
		mask.append(index)
	return mask


def BestAntMask(x, y):
	buckets = CreateFeatureBuckets(len(x[0]))
	n = 0
	while True:
		n += 1

		ants = CreateAnts(buckets)
		fit = EvaluatePopulation(ants, buckets, x, y)

		if n >= NIter:
			break
		ApplyDecay(buckets)
		ApplyPheromones(buckets, ants, fit)

		accuracy = EvaluateMask(DeriveMask(buckets), x, y)
		print("accuracy = "+str(accuracy))


	return DeriveMask(buckets)

x, y = LoadFeatures(all_features_file_name)

mask = BestAntMask(x, y)

#print(mask)

accuracy = EvaluateMask(mask, x, y, feature_weight=0)
print("accuracy = "+str(accuracy))


