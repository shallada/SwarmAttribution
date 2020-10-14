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
		total_accuracy = 0.0
		total_percent_ones = 0.0
		avg_mask = None
		best_acc = -1.0
		best_mask = None
		cumulative_mask = None
		for n in range(NRuns):
			run_no = n+1
			in_file_name = folder+"/"+algo+"-"+data_set+"-"+str(ones_ratio)+"-"+str(run_no)
			with open(in_file_name, "r") as in_file:
				line = in_file.read()
				accuracy = float(line.split(',')[0])
				mask_line = line.split('[')[1][:-2]
				mask = []
				for s in mask_line.split(','):
					mask.append(int(s))
				if cumulative_mask == None:
					cumulative_mask = np.zeros([len(mask),])
				cumulative_mask += mask
				percent_ones = sum(mask)/len(mask)
				if accuracy > best_acc:
					best_acc = accuracy
					best_mask = percent_ones
				total_accuracy += accuracy
				total_percent_ones += percent_ones
		cumulative_mask /= float(NRuns)
		s = ""
		for x in cumulative_mask:
			s += str(x) + ", "
		print(algo+", "+str(ones_ratio)+", "+s[:-2])
		#print(s[:-2])




