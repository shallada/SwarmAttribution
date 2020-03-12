import csv
import numpy as np
import random
import re
import sys
from sklearn import preprocessing

from EvaluateMask import EvaluateMask

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

"""
def EvaluateMask(mask, x, y):
	kfold = StratifiedKFold(n_splits=4,shuffle=True,random_state=0)


	#
	# Adjust the pipeline to select the desired feature preprocessing, preprocessing order and
	# Author attribution kernel.
	#
	# Note TF-IDF is not working - probably needs CountVectorizer
	#
	pipeline = Pipeline([
		#('tfidf', TfidfTransformer()),
		('standardizer', StandardScaler()),
		('normalizer', Normalizer()),
		('clf', OneVsRestClassifier(svm.SVC(kernel='linear'),n_jobs=-1))
		#('clf', OneVsRestClassifier(svm.SVC(kernel='rbf', gamma='auto'),n_jobs=-1))
		#('mlp', MLPClassifier(hidden_layer_sizes=(100), max_iter=10000, activation = 'relu', solver='adam'))
	])

	fold_accuracy = []
	for train, test in StratifiedKFold(n_splits=4).split(x, y):
		x_train = np.array(x[train]) * np.array(mask)
		y_train = np.array(y[train])

		x_test = np.array(x[test]) * np.array(mask)
		y_test = np.array(y[test])


		pipeline.fit(x_train, y_train)
		accuracy = pipeline.score(x_test, y_test)

		fold_accuracy.append(accuracy)
	return np.mean(fold_accuracy)
"""

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


