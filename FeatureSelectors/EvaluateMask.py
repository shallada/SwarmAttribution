import numpy as np
from scipy import sparse
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer


FeatureWeight = 0.0

def CalculateTF_IDF(matrix):
	matrix = sparse.csr_matrix(matrix)
	tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
	tfidf_transformer.fit(matrix)
	matrix = tfidf_transformer.transform(matrix)
	matrix = matrix.todense()
	return matrix


def EvaluateMask(mask, x, y, feature_weight=FeatureWeight):

	kfold = StratifiedKFold(n_splits=4,shuffle=True,random_state=0)


	#
	# Adjust the pipeline to select the desired feature preprocessing, preprocessing order and
	# Author attribution kernel.
	#
	# Note TF-IDF is not working - probably needs CountVectorizer
	#
	pipeline = Pipeline([
		('standardizer', StandardScaler()),
		('normalizer', Normalizer()),

		# Choose one of the following:

		# Support Vector Machine
		#('clf', OneVsRestClassifier(svm.SVC(kernel='linear'),n_jobs=-1))

		# Radial Basis Function
		('clf', OneVsRestClassifier(svm.SVC(kernel='rbf', gamma='auto'),n_jobs=-1))

		# Multi-level perceptron (Simple neural net)
		#('mlp', MLPClassifier(hidden_layer_sizes=(100), max_iter=10000, activation = 'relu', solver='adam'))
	])

	fold_fitness = []
	mask_array = np.array(mask)
	mask_sum = np.sum(mask_array)
	for train, test in StratifiedKFold(n_splits=4).split(x, y):
		x_train = np.array(x[train]) * mask_array
		y_train = np.array(y[train])

		x_test = np.array(x[test]) * mask_array
		y_test = np.array(y[test])

		x_train = CalculateTF_IDF(x_train)
		x_test = CalculateTF_IDF(x_test)

		pipeline.fit(x_train, y_train)
		accuracy = pipeline.score(x_test, y_test)
		fitness = accuracy - (feature_weight * (mask_sum/len(mask)))

		fold_fitness.append(fitness)

	return np.mean(fold_fitness)

