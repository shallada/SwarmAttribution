import Data_Utils
from Extractor.DatasetInfo import DatasetInfo
from Extractor.Extractors import BagOfWords, Stylomerty, Unigram, CharacterGram
import sys
import numpy as np



if len(sys.argv) != 2:
	print("Missing parameter data_set")
	exit()

data_set = sys.argv[1]

data_dir = "../../Datasets/"
feature_set_dir = "./datasets/"
feature_file = "../../Concatenator/"+data_set+"/StylometryFeatures.txt"

if __name__ == "__main__":
    #  extractor = Unigram(data_dir + "CASIS25/", "casis25")
    #  extractor = Stylomerty(data_dir + "CASIS25/", "casis25")
    #  extractor = BagOfWords(data_dir + "CASIS25/", "casis25")
    extractor = CharacterGram(data_dir + data_set+ "/", data_set, gram=3, limit=1000)
    extractor.start()
    lookup_table = extractor.lookup_table
    print("Generated Lookup Table:")
    print(lookup_table)
    if lookup_table is not False:
        print("'"+"', '".join([str("".join(x)).replace("\n", " ") for x in lookup_table]) + "'")

    # Get dataset information
    dataset_info = DatasetInfo(data_set+"_bow")
    dataset_info.read()
    authors = dataset_info.authors
    writing_samples = dataset_info.instances
    print("\n\nAuthors in the dataset:")
    print(authors)

    print("\n\nWriting samples of an author 1000")
    print(authors["1000"])

    print("\n\nAll writing samples in the dataset")
    print(writing_samples)

    print("\n\nThe author of the writing sample 1000_1")
    print(writing_samples["1000_1"])

    generated_file = feature_set_dir + extractor.out_file + ".txt"
    data, labels = Data_Utils.get_dataset(generated_file)
    print(labels[0], data[0])
    features = []
    for i in range(len(labels)):
        if len(labels[i]) > 0:
	        r = [labels[i]+".txt"]
	        for val in data[i]:
	            r.append(val)
	        features.append(r)
    features = sorted(features)
    with open(feature_file, "w") as out_file:
        for i in range(len(features)):
            l = features[i][0]
            for j in range(1, len(features[i])):
                l += "," + str(int(features[i][j]))
            out_file.write(l+"\n")
