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


def RandomDiscreteMask(mask_size, ones_ratio):
	mask = []
	for _ in range(mask_size):
		mask.append(1 if random.random() < ones_ratio else 0)
	return mask



x, y = LoadFeatures(all_features_file_name)


""" Generate random features just to see what happens
ranx = []
for _ in range(len(x)):
	ranx.append([random.random() for __ in range(len(x[0]))])
x = np.array(ranx)
"""

mask_size = len(x[0])




""" Loop to increase the number of features
for n in range(0, 110, 10):
	ratio = n/100.0
	mask = RandomDiscreteMask(mask_size, ratio)
	accuracy = EvaluateMask(mask, x, y, feature_weight=0)
	print("ratio = "+str(ratio)+", accuracy = "+str(accuracy))
"""



"""
# random mask
mask = [random.random() for j in range(mask_size)]
accuracy = EvaluateMask(mask, x, y, feature_weight=0)
print("accuracy = "+str(accuracy))

# all ones mask
mask = np.ones(mask_size)
accuracy = EvaluateMask(mask, x, y, feature_weight=0)
print("accuracy = "+str(accuracy))

#all zeros mask
mask = np.zeros(mask_size)
accuracy = EvaluateMask(mask, x, y, feature_weight=0)
print("accuracy = "+str(accuracy))

# random discrete mask
mask = np.random.choice([0, 1], size=(mask_size,)).tolist()
accuracy = EvaluateMask(mask, x, y, feature_weight=0)
print("accuracy = "+str(accuracy))
"""



# Shift a zero through the mask, then turn off the bad features
#mask = np.ones(mask_size)
mask = [random.random() for j in range(mask_size)]
#mask = np.zeros(mask_size)

accuracies = []
for i in range(mask_size):
	save = mask[i]
	if mask[i] > 0.0:
		mask[i] = 0
	accuracy = EvaluateMask(mask, x, y, feature_weight=0)
	mask[i] = save
	print(str(i)+": accuracy = "+str(accuracy))
	accuracies.append(accuracy)

mean = sum(accuracies) / len(accuracies)
print("mean = "+str(mean))
mask = list(map(lambda x: 0 if x > mean else 1, accuracies))
print("sum(mask) = "+str(sum(mask)))

accuracy = EvaluateMask(mask, x, y, feature_weight=0)
print("accuracy = "+str(accuracy))

accuracies = []
for i in range(mask_size):
	save = mask[i]
	if mask[i] > 0.0:
		mask[i] = 0
	accuracy = EvaluateMask(mask, x, y, feature_weight=0)
	mask[i] = save
	print(str(i)+": accuracy = "+str(accuracy))
	accuracies.append(accuracy)

mean = sum(accuracies) / len(accuracies)
print("mean = "+str(mean))
mask = list(map(lambda x: 0 if x > .87 else 1, accuracies))
print("sum(mask) = "+str(sum(mask)))

accuracy = EvaluateMask(mask, x, y, feature_weight=0)
print("accuracy = "+str(accuracy))

