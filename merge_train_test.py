import csv
import pandas as pd
import os, pickle

#train
dataframes = []
directory = 'data/train'
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), 'rb') as f:
        data = pickle.load(f)
        seizure = data[data.seizure == 1]
        non_seizure = data[data.seizure == 0]
        non_seizure = non_seizure.sample(seizure.shape[0])
        data = pd.concat([seizure, non_seizure]).sample(frac=1)
        dataframes.append(data)

dataframes = pd.concat(dataframes).reset_index().drop(columns=["index"])
with open('data/train.pickle', 'wb') as f:
    pickle.dump(dataframes, f)

#test                                                                                           
dataframes = []
directory = 'data/test'
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), 'rb') as f:
        data = pickle.load(f)
        seizure = data[data.seizure == 1]
        non_seizure = data[data.seizure == 0]
        non_seizure = non_seizure.sample(seizure.shape[0])
        data = pd.concat([seizure, non_seizure]).sample(frac=1)
        dataframes.append(data)

dataframes = pd.concat(dataframes).reset_index().drop(columns=["index"])
with open('data/test.pickle', 'wb') as f:
    pickle.dump(dataframes, f)