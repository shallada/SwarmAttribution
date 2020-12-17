import os
import sys
import pdb

file_dir = "../../Datasets/"+sys.argv[1]+"/"
results_dir = "../../Concatenator/"+sys.argv[1]+"/OpinionFinderWork/"

output_file_name = "../../Concatenator/"+sys.argv[1]+"/SentimentAnalysisFeatures.txt"
if os.path.exists(output_file_name):
	os.remove(output_file_name)

files = []
for filename in os.listdir(file_dir):
    if "_auto_anns" not in filename:
        files.append(filename)

instance = 0
for raw_filename in sorted(files):
    filename = results_dir + "{}_auto_anns/subjclueslen1polar".format(raw_filename)
    transistion_matrix = [[0 for _ in range(12)] for _ in range(12)]
    transition_matrix_Second_Order = [[[0 for _ in range(12)] for _ in range(12)] for _ in range(12)]
    old_state = -1
    previous_state = -1
    current_state = -1

    num_strongsubj = 0.0
    num_weaksubj = 0.0
    num_strongpos = 0.0
    num_weakpos = 0.0
    num_neutral = 0.0
    num_weakneg = 0.0
    num_strongneg = 0.0
    num_both = 0.0
    total = 0.0

    strongsubj_strongpos = 0
    strongsubj_weakpos = 0
    strongsubj_neutral = 0
    strongsubj_weakneg = 0
    strongsubj_strongneg = 0
    strongsubj_both = 0

    weaksubj_strongpos = 0
    weaksubj_weakpos = 0
    weaksubj_neutral = 0
    weaksubj_weakneg = 0
    weaksubj_strongneg = 0
    weaksubj_both = 0

    with open(filename, "r") as file:
        for line in file:
            strong = 0
            total = total + 1
            components = line.split("\t")
            if "weaksubj" in components[3]:
                num_weaksubj += 1
                strong = 0
            elif "strongsubj" in components[3]:
                num_strongsubj += 1
                strong = 1
            else:
                raise ValueError("broke")

            if "strongpos" in components[3]:
                num_strongpos += 1
                if strong == 1:
                    strongsubj_strongpos += 1
                    current_state = 0
                else:
                    weaksubj_strongpos += 1
                    current_state = 6
            elif "weakpos" in components[3]:
                num_weakpos += 1
                if strong == 1:
                    strongsubj_weakpos += 1
                    current_state = 1
                else:
                    weaksubj_weakpos += 1
                    current_state = 7
            elif "neutral" in components[3]:
                num_neutral += 1
                if strong == 1:
                    strongsubj_neutral += 1
                    current_state = 2
                else:
                    weaksubj_neutral += 1
                    current_state = 8
            elif "weakneg" in components[3]:
                num_weakneg += 1
                if strong == 1:
                    strongsubj_weakneg += 1
                    current_state = 3
                else:
                    weaksubj_weakneg += 1
                    current_state = 9
            elif "strongneg" in components[3]:
                num_strongneg += 1
                if strong == 1:
                    strongsubj_strongneg += 1
                    current_state = 4
                else:
                    weaksubj_strongneg += 1
                    current_state = 10
            elif "both" in components[3]:
                num_both += 1
                if strong == 1:
                    strongsubj_both += 1
                    current_state = 5
                else:
                    weaksubj_both += 1
                    current_state = 11
            else:
                raise ValueError("broke")
            if previous_state != -1:
                transistion_matrix[previous_state][current_state] += 1
            if old_state != -1:
                transition_matrix_Second_Order[old_state][previous_state][current_state] += 1
            old_state = previous_state
            previous_state = current_state
    
    if total == 0:
        with open(output_file_name, "a") as file:
            file.write("{},{}".format(raw_filename, ",".join(["0" for _ in range(176)])))
        continue
    prob_strongsubj = num_strongsubj/total
    prob_weaksubj = num_weaksubj/total

    prob_strongpos = num_strongpos/total
    prob_weakpos = num_weakpos/total
    prob_neutral = num_neutral/total
    prob_weakneg = num_weakneg/total
    prob_strongneg = num_strongneg/total
    prob_both = num_both/total

    prob_strongsubj_strongpos = strongsubj_strongpos/total
    prob_strongsubj_weakpos = strongsubj_weakpos/total
    prob_strongsubj_neutral = strongsubj_neutral/total
    prob_strongsubj_weakneg = strongsubj_weakneg/total
    prob_strongsubj_strongneg = strongsubj_strongneg/total
    prob_strongsubj_both = strongsubj_both/total

    prob_weaksubj_strongpos = weaksubj_strongpos/total
    prob_weaksubj_weakpos = weaksubj_weakpos/total
    prob_weaksubj_neutral = weaksubj_neutral/total
    prob_weaksubj_weakneg = weaksubj_weakneg/total
    prob_weaksubj_strongneg = weaksubj_strongneg/total
    prob_weaksubj_both = weaksubj_both/total

    features = []
    #features.append(num_strongsubj)
    #features.append(num_weaksubj)

    #features.append(num_strongpos)
    #features.append(num_weakpos)
    #features.append(num_neutral)
    #features.append(num_weakneg)
    #features.append(num_strongneg)
    #features.append(num_both)

    #features.append(strongsubj_strongpos)
    #features.append(strongsubj_weakpos)
    #features.append(strongsubj_neutral)
    #features.append(strongsubj_weakneg)
    #features.append(strongsubj_strongneg)
    #features.append(strongsubj_both)

    #features.append(weaksubj_strongpos)
    #features.append(weaksubj_weakpos)
    #features.append(weaksubj_neutral)
    #features.append(weaksubj_weakneg)
    #features.append(weaksubj_strongneg)
    #features.append(weaksubj_both)

    #features.append(total)

    # Pr(subjectivity|polarity)
    features.append(prob_strongsubj)
    features.append(prob_weaksubj)

    features.append(prob_strongpos)
    features.append(prob_weakpos)
    features.append(prob_neutral)
    features.append(prob_weakneg)
    features.append(prob_strongneg)
    features.append(prob_both)

    features.append((prob_strongsubj_strongpos/prob_strongpos) if prob_strongpos > 0 else 0)
    features.append((prob_strongsubj_weakpos/prob_weakpos) if prob_weakpos > 0 else 0)
    features.append((prob_strongsubj_neutral/prob_neutral) if prob_neutral > 0 else 0)
    features.append((prob_strongsubj_weakneg/prob_weakneg) if prob_weakneg > 0 else 0)
    features.append((prob_strongsubj_strongneg/prob_strongneg) if prob_strongneg > 0 else 0)
    features.append((prob_strongsubj_both/prob_both) if prob_both > 0 else 0)
    features.append((prob_weaksubj_strongpos/prob_strongpos) if prob_strongpos > 0 else 0)
    features.append((prob_weaksubj_weakpos/prob_weakpos) if prob_weakpos > 0 else 0)
    features.append((prob_weaksubj_neutral/prob_neutral) if prob_neutral > 0 else 0)
    features.append((prob_weaksubj_weakneg/prob_weakneg) if prob_weakneg > 0 else 0)
    features.append((prob_weaksubj_strongneg/prob_strongneg) if prob_strongneg > 0 else 0)
    features.append((prob_weaksubj_both/prob_both) if prob_both > 0 else 0)

    features.append((prob_strongsubj_strongpos/prob_strongsubj) if prob_strongsubj > 0 else 0)
    features.append((prob_strongsubj_weakpos/prob_strongsubj) if prob_strongsubj > 0 else 0)
    features.append((prob_strongsubj_neutral/prob_strongsubj) if prob_strongsubj > 0 else 0)
    features.append((prob_strongsubj_weakneg/prob_strongsubj) if prob_strongsubj > 0 else 0)
    features.append((prob_strongsubj_strongneg/prob_strongsubj) if prob_strongsubj > 0 else 0)
    features.append((prob_strongsubj_both/prob_strongsubj) if prob_strongsubj > 0 else 0)
    features.append((prob_weaksubj_strongpos/prob_weaksubj) if prob_weaksubj > 0 else 0)
    features.append((prob_weaksubj_weakpos/prob_weaksubj) if prob_weaksubj > 0 else 0)
    features.append((prob_weaksubj_neutral/prob_weaksubj) if prob_weaksubj > 0 else 0)
    features.append((prob_weaksubj_weakneg/prob_weaksubj) if prob_weaksubj > 0 else 0)
    features.append((prob_weaksubj_strongneg/prob_weaksubj) if prob_weaksubj > 0 else 0)
    features.append((prob_weaksubj_both/prob_weaksubj) if prob_weaksubj > 0 else 0)

    #pdb.set_trace()
    with open(output_file_name, "a") as file:
        sentiment_transitions = ""
        first = 0
        for x in transistion_matrix:
            if first == 0:
                first += 1
                sentiment_transitions = ",".join(str(f) for f in x)
            else:
                sentiment_transitions += "," + ",".join(str(f) for f in x)
        # for x in range(12):
            # for y in range(12):
                # for z in range(12):
                    # sentiment_transitions += "," + str(transition_matrix_Second_Order[x][y][z])
        file.write("{},{},{}\n".format(raw_filename,",".join(str(x) for x in features), sentiment_transitions))
        instance = instance + 1




