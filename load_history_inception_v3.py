import pickle
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

base_path = 'inception_v3_history/'


training_history_files = [f for f in listdir(base_path) if isfile(join(base_path, f)) and 'trainHistoryDict' in f]


max_allowed_loss = 0.5
min_allowed_accuracy = 0.82

histories_found = {}

for training_history_file in training_history_files:
    with open('inception_v3_history/' + training_history_file, 'rb') as handle:
        training_history = pickle.load(handle)
        criteria_match_indices = []


        for idx, (val_loss, val_acc) in enumerate(zip(training_history['val_loss'], training_history['val_acc'])):
            
            loss_criterion_met = (val_loss < max_allowed_loss)
            accuracy_criterion_met = (val_acc > min_allowed_accuracy)

            if accuracy_criterion_met:
                criteria_match_indices.append(idx)

        if criteria_match_indices != []:
            histories_found[training_history_file.split('.')[0].split('_')[1]] = criteria_match_indices

if histories_found != {}:
    print("*************************")
    print("The specified criteria were met for the following hyperparameter values: ")

    for hyp_val, criteria_match_indices in histories_found.items():
        with open('inception_v3_history/trainHistoryDict_' + hyp_val + '.pickle', 'rb') as handle:
            training_history = pickle.load(handle)

        print("     For number of non-trainable layers = " + hyp_val + ":")

        for idx in criteria_match_indices:
            print("       - at epoch " + str(idx+1) + ", validation loss was " + str(round(training_history['val_loss'][idx], 3)) + "; validation accuracy was " + str(round(training_history['val_acc'][idx], 3)))
else:
    print("None of the hyper parameter values could meet the specified criteria.")



hyp_val_criteria_occurences = [(hyp_val, len(criteria_match_indices)) for (hyp_val, criteria_match_indices) in histories_found.items()]
hyp_vals, num_criteria_occurences = list(zip(*hyp_val_criteria_occurences))

x = range(1, 101)
y = []
for i in x:
    if str(i) in histories_found.keys():
        y.append(len(histories_found[str(i)]))
    else:
        y.append(0)



plt.bar(x=x, height=y)
plt.xlabel('Number of Trainable Layers')
plt.ylabel('Number of Criteria Matches')
plt.show()