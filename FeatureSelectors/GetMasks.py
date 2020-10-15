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
			print(algo + ", " + str(ones_ratio) + ", " + str(n) + ", " + mask_line)



