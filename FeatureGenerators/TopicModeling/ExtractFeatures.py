import pdb
import sys

input_file_name = "../../Concatenator/"+sys.argv[1]+"/MalletWork/Topics.mallet"
output_file_name = "../../Concatenator/"+sys.argv[1]+"/TopicModelingFeatures.txt"

data = []
with open(input_file_name, "r") as features_file:
	#next(features_file)

	for line in features_file:
		line = line.strip().split("\t")
		label = line[1].split("/")[-1]
		#doc_num = line[1].split("/")[-1][:6]
		features = [str(x) for x in line[2:]]
		#data.append((label, doc_num, features))
		data.append((label, features))

data = sorted(data, key=lambda x: (x[0], x[1]))
with open(output_file_name, "w") as output_file:
	#for label, _, features in data:
	for label, features in data:
		output_file.write("{},{}\n".format(label, ",".join(features)))
