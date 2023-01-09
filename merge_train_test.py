import csv
import pandas as pd
import os, pickle

#train
dataframes = []
directory = 'Code\\data\\train'
for filename in os.listdir(directory):
    data = pd.read_csv(os.path.join(directory, filename))
    seizure = data[data.seizure == 1]
    non_seizure = data[data.seizure == 0]
    non_seizure = non_seizure.sample(seizure.shape[0])
    data = pd.concat([seizure, non_seizure]).sample(frac=1)
    dataframes.append(data)

dataframes = pd.concat(dataframes).reset_index().drop(columns=["index"])
with open('Code\\data\\train.pickle', 'wb') as f:
    pickle.dump(dataframes, f)

#test
dataframes = []
directory = 'Code\\data\\test'
for filename in os.listdir(directory):
    data = pd.read_csv(os.path.join(directory, filename))
    seizure = data[data.seizure == 1]
    non_seizure = data[data.seizure == 0]
    non_seizure = non_seizure.sample(seizure.shape[0])
    data = pd.concat([seizure, non_seizure]).sample(frac=1)
    dataframes.append(data)

dataframes = pd.concat(dataframes).reset_index().drop(columns=["index"])
with open('Code\\data\\test.pickle', 'wb') as f:
    pickle.dump(dataframes, f)