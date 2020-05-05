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
NRuns = 30

def RandomDiscreteMask(mask_size, ones_ratio):
	mask = []
	for _ in range(mask_size):
		mask.append(1 if random.random() < ones_ratio else 0)
	return mask



x, y = LoadFeatures(all_features_file_name)

mask_size = len(x[0])

for ones_ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
	for n in range(NRuns):
		best_accuracy = 0.0
		for _ in range(500):
			mask = RandomDiscreteMask(mask_size, ones_ratio)
			accuracy = EvaluateMask(mask, x, y, feature_weight=0)
			if accuracy > best_accuracy:
				best_accuracy = accuracy

		accuracy = best_accuracy
		print(str(n)+": ones_ratio = "+str(ones_ratio)+", accuracy = "+str(accuracy), flush = True)



