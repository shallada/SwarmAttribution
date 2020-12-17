import os
import sys

dataset_dir = "../../Datasets/"+sys.argv[1]+"/"
filenames = os.listdir(dataset_dir)
with open("../../Concatenator/"+sys.argv[1]+"/OpinionFinderWork/SentimentAnalysis.doclist", "w") as doclist_file:
    for filename in filenames:
        if ".txt" in filename:
            doclist_file.write("{}{}\n".format(dataset_dir, filename))

