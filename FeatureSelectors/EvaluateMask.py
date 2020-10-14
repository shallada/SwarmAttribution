import numpy as np
from scipy import sparse
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, normalize

LATE_PIPE_MERGE = True
N_LIWC_FEATURES = 93
N_SA_FEATURES = 176
N_TM_FEATURES = 45

FeatureWeight = 0.0


def EvaluateMask(mask, x, y, feature_weight=FeatureWeight):

	kfold = StratifiedKFold(n_splits=4,shuffle=True,random_state=0)


	#
	# Adjust the pipeline to select the desired feature preprocessing, preprocessing order and
	# Author attribution kernel.
	#
	# Note TF-IDF is not working - probably needs CountVectorizer
	#
	pipeline = Pipeline([
		#('standardizer', StandardScaler()),
		#('normalizer', Normalizer()),

		# Choose one of the following:

		# Support Vector Machine
		#('clf', OneVsRestClassifier(svm.SVC(kernel='linear'),n_jobs=-1))

		# Radial Basis Function
		('clf', OneVsRestClassifier(svm.SVC(kernel='rbf', gamma='auto'),n_jobs=-1))

		# Multi-level perceptron (Simple neural net)
		('mlp', MLPClassifier(hidden_layer_sizes=(100), max_iter=10000, activation = 'relu', solver='adam'))
	])

	fold_fitness = []
	mask_array = np.array(mask)
	mask_sum = np.sum(mask_array)
	for train, test in StratifiedKFold(n_splits=4).split(x, y):
		x_train = np.array(x[train]) * mask_array
		y_train = np.array(y[train])

		x_test = np.array(x[test]) * mask_array
		y_test = np.array(y[test])


		if LATE_PIPE_MERGE:
			# split up the features prior to preprocessing
			x_train_liwc = x_train[:, :N_LIWC_FEATURES]
			x_train_sa = x_train[:, N_LIWC_FEATURES: N_LIWC_FEATURES+N_SA_FEATURES]
			x_train_tm = x_train[:, N_LIWC_FEATURES+N_SA_FEATURES:]

			x_test_liwc = x_test[:, :N_LIWC_FEATURES]
			x_test_sa = x_test[:, N_LIWC_FEATURES: N_LIWC_FEATURES+N_SA_FEATURES]
			x_test_tm = x_test[:, N_LIWC_FEATURES+N_SA_FEATURES:]

			# perform TFIDF procesing
			x_train_liwc = sparse.csr_matrix(x_train_liwc)
			x_test_liwc = sparse.csr_matrix(x_test_liwc)
			tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
			tfidf_transformer.fit(x_train_liwc)
			x_train_liwc = tfidf_transformer.transform(x_train_liwc)
			x_test_liwc = tfidf_transformer.transform(x_test_liwc)
			x_train_liwc = x_train_liwc.todense()
			x_test_liwc = x_test_liwc.todense()

			x_train_sa = sparse.csr_matrix(x_train_sa)
			x_test_sa = sparse.csr_matrix(x_test_sa)
			tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
			tfidf_transformer.fit(x_train_sa)
			x_train_sa = tfidf_transformer.transform(x_train_sa)
			x_test_sa = tfidf_transformer.transform(x_test_sa)
			x_train_sa = x_train_sa.todense()
			x_test_sa = x_test_sa.todense()

			x_train_tm = sparse.csr_matrix(x_train_tm)
			x_test_tm = sparse.csr_matrix(x_test_tm)
			tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
			tfidf_transformer.fit(x_train_tm)
			x_train_tm = tfidf_transformer.transform(x_train_tm)
			x_test_tm = tfidf_transformer.transform(x_test_tm)
			x_train_tm = x_train_tm.todense()
			x_test_tm = x_test_tm.todense()

			# perform standardization processing
			scaler_liwc = StandardScaler()
			scaler_sa = StandardScaler()
			scaler_tm = StandardScaler()

			scaler_liwc.fit(x_train_liwc)
			scaler_sa.fit(x_train_sa)
			scaler_tm.fit(x_train_tm)

			x_train_liwc = scaler_liwc.transform(x_train_liwc)
			x_train_sa = scaler_sa.transform(x_train_sa)
			x_train_tm = scaler_tm.transform(x_train_tm)

			x_test_liwc = scaler_liwc.transform(x_test_liwc)
			x_test_sa = scaler_sa.transform(x_test_sa)
			x_test_tm = scaler_tm.transform(x_test_tm)
			
			# perform normalization processing
			x_train_liwc = normalize(x_train_liwc)
			x_train_sa = normalize(x_train_sa)
			x_train_tm = normalize(x_train_tm)

			x_test_liwc = normalize(x_test_liwc)
			x_test_sa = normalize(x_test_sa)
			x_test_tm = normalize(x_test_tm)

			# recombine features
			x_train = np.concatenate((x_train_liwc, x_train_sa, x_train_tm), axis=1)
			x_test = np.concatenate((x_test_liwc, x_test_sa, x_test_tm), axis=1)
		else:
			# keep all features together for preprocessing
			# perform TFIDF procesing
			x_train = sparse.csr_matrix(x_train)
			x_test = sparse.csr_matrix(x_test)
			tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
			tfidf_transformer.fit(x_train)
			x_train = tfidf_transformer.transform(x_train)
			x_test = tfidf_transformer.transform(x_test)
			x_train = x_train.todense()
			x_test = x_test.todense()

			# perform standardization processing
			scaler = StandardScaler()
			scaler.fit(x_train)
			x_train = scaler.transform(x_train)
			x_test = scaler.transform(x_test)
			
			# perform normalization processing
			x_train = normalize(x_train)
			x_test = normalize(x_test)

		pipeline.fit(x_train, y_train)
		accuracy = pipeline.score(x_test, y_test)
		fitness = accuracy - (feature_weight * (mask_sum/len(mask)))

		fold_fitness.append(fitness)

	return np.mean(fold_fitness)

