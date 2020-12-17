import csv
import numpy as np
import sys


features = []
def merge_features(liwc_file_name, tm_file_name, sty_file_name, output_file_name):
	with open(liwc_file_name, "r") as liwc_file:
		with open(tm_file_name, "r") as tm_file:
			with open(sty_file_name, "r") as sty_file:
				for liwc_line, tm_line, sty_line in zip(liwc_file, tm_file, sty_file):
					liwc_line = liwc_line.strip().split(",")
					tm_line = tm_line.strip().split(",")
					sty_line = sty_line.strip().split(",")
					assert liwc_line[0] == tm_line[0]
					assert sty_line[0] == liwc_line[0]

					# note that LIWC skips 2 - the label and useless segment that is always 1
					line = [liwc_line[0]] + [float(x) for x in liwc_line[2:]] + [float(x) for x in tm_line[1:]] + [float(x) for x in sty_line[1:]]
					features.append(line)
	with open(output_file_name, "w") as out_file:
		writer = csv.writer(out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		writer.writerows(features)

merge_features(sys.argv[1]+"/LIWCFeatures.txt", sys.argv[1]+"/TopicModelingFeatures.txt", sys.argv[1]+"/StylometryFeatures.txt", sys.argv[1]+"/AllFeatures.txt")
