import csv
import re
import numpy as np
from sklearn import preprocessing

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
