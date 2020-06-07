import math
import numpy as np
import sys


if len(sys.argv) != 3:
        print("Missing parameters")
        print("FORMAT: data_set folder")
        exit()

data_set = sys.argv[1]
folder = sys.argv[2]

NRuns = 30

for algo in ["RandomMask", "ABCFeatureSelection", "GlowwormSwarmOptimization", "ParticleSwarmSelection", "AntSystemSelection", "EvolveFeatureSelection"]:
	for ones_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
		if algo == "RandomMask" and ones_ratio == 0.0:
			continue
		total_accuracy = 0.0
		avg_mask = None
		for n in range(NRuns):
			run_no = n+1
			in_file_name = folder+"/"+algo+"-"+data_set+"-"+str(ones_ratio)+"-"+str(run_no)
			with open(in_file_name, "r") as in_file:
				line = in_file.read()
				accuracy = float(line.split(',')[0])
				total_accuracy += accuracy
				mask_line = line.split('[')[1][:-2]
				mask = []
				for s in mask_line.split(','):
					mask.append(int(s))
				if avg_mask == None:
					avg_mask = mask
				else:
					avg_mask = [sum(i) for i in zip(avg_mask, mask)]
		print(algo+", "+str(ones_ratio)+", "+str(round(100*total_accuracy/NRuns, 2)))
		#print(avg_mask)




