Swarm Feature Selection
-----------------------

The Swarm Feature Selection system works as a pipeline. The system feeds datasets into feature generators which identify features in the text. For each document in each dataset, the concatenator combines the features into a single vector. Next, the feature selection mechanisms consume the feature vector and output a weighted feature vector. Finally, the attribution kernels use the weighted feature vectors to predict the author.


