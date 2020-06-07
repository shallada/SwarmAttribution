import math
import numpy as np
import random
import sys



from EvaluateMask import EvaluateMask
from LoadFeatures import LoadFeatures



if len(sys.argv) != 4:
	print("Missing parameters")
	print("FORMAT: algorith data_set ones_ratio run")
	exit()

# Don't run with no ones
if sys.argv[2] == "0.0":
	exit()

algorithm = sys.argv[0].split(".")[0]
data_set = sys.argv[1]
ones_ratio = float(sys.argv[2])
run_no = int(sys.argv[3])
all_features_file_name = "../Concatenator/"+data_set+"/AllFeatures.txt"
out_file_name = "output/"+algorithm+"-"+data_set+"-"+str(ones_ratio)+"-"+str(run_no)



NEvals = 1500

def RandomDiscreteMask(mask_size, ones_ratio):
	mask = []
	for _ in range(mask_size):
		mask.append(1 if random.random() < ones_ratio else 0)
	return mask



x, y = LoadFeatures(all_features_file_name)

mask_size = len(x[0])

with open(out_file_name, 'w') as out_file:

	best_accuracy = 0.0
	for _ in range(NEvals):
		mask = RandomDiscreteMask(mask_size, ones_ratio)
		accuracy = EvaluateMask(mask, x, y, feature_weight=0)
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_mask = mask

	accuracy = best_accuracy

	out_file.write(str(accuracy)+","+str(best_mask)+"\n")
