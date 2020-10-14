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

#algo_list = ["RandomMask", "ABCFeatureSelection", "GlowwormSwarmOptimization", "ParticleSwarmSelection", "AntSystemSelection", "EvolveFeatureSelection"]
algo_list = ["EvolveFeatureSelection"]
ones_ratio_list = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


all_results = np.zeros((NRuns, len(ones_ratio_list)))
for algo in algo_list:
	col = 0
	for ones_ratio in ones_ratio_list:
		for n in range(NRuns):
			run_no = n+1
			in_file_name = folder+"/"+algo+"-"+data_set+"-"+str(ones_ratio)+"-"+str(run_no)
			with open(in_file_name, "r") as in_file:
				line = in_file.read()
				accuracy = float(line.split(',')[0])
			all_results[n, col] = accuracy
		col += 1
	for row in range(len(all_results)):
		row_str = ""
		for col in range(len(all_results[0])):
			row_str += ", " + str(all_results[row, col]) 
		print(algo + row_str)




