import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
import csv, mne, math, pickle, json
from IPython.display import clear_output

# constants
EPOCH_DURATION = 2
OVERLAP_DURATION = 1

# randomly selected patients to be train or test
train = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb10",
    "chb11",
    "chb13",
    "chb14",
    "chb15",
    "chb16",
    "chb17",
    "chb18",
    "chb19",
    "chb20",
    "chb21",
    "chb22",
    "chb23",
    "chb24",
]

test = ["chb07", "chb08", "chb09"]

# drive path to the dataset (should change it if you want to try the code)
drive_path = "C:/Users/Eugene Chen/Desktop/UNI/Project/Data/"
seizure_pointers = pd.read_excel(drive_path + "seizure data.xlsx", index_col=0)
seizure_pointers["index"] = (
    seizure_pointers["seizure_file"]
    + " "
    + seizure_pointers["seizure_number"].astype(str)
)

# channels to be selected
seizure_pointers = seizure_pointers.set_index("index")
channels = [
    "P8-O2",
    "C4-P4",
    "FP1-F3",
    "FP2-F8",
    "CZ-PZ",
    "FP1-F7",
    "T7-P7",
    "C3-P3",
    "FP2-F4",
    "P4-O2",
    "F8-T8",
    "F7-T7",
    "F3-C3",
    "FZ-CZ",
    "P3-O1",
    "P7-O1",
    "F4-C4",
]

# labels to identify the features of output dataset
label_std = [n + "-std" for n in channels]
label_var = [n + "-var" for n in channels]
label_max = [n + "-max" for n in channels]
label_std_rfft = [n + "-std_rfft" for n in channels]
label_var_rfft = [n + "-var_rfft" for n in channels]
label_max_rfft = [n + "-max_rfft" for n in channels]


# only for a loading bar can be ignored
DONE = 1
TOTAL = 0
for t in train + test:
    TOTAL += seizure_pointers[seizure_pointers.case == t].shape[0]

# loading bar
def progress_bar():
    percent = 100 * (DONE / float(TOTAL))
    bar = "█" * int(percent) + "-" * (100 - int(percent))
    print(f"\r|{bar}| {percent: .2f}%", end="\r")

def diff(lst1, lst2):
    '''
    input:
    lst1 -> list 
    lst2 -> list
    output:
    return intersection of lst1 with lst2
    '''
    return list(set(lst1) - set(lst2))

def get_reduced_freq(target, batch_size, sampling_rate):
    '''
    input:
    target -> sample to apply the tranformation
    batch_size -> lenght of the signal reduction (int)
    sampling_rate -> sampling rate of the signal (int)
    output:
    return target applied reduction and fourier tranformation
    '''
    result = []
    for channel in target:
        layer = []
        batch=[]
        target_ft = abs(rfft(channel))
        target_ft = [ x.real for x in target_ft]
        for i in target_ft[0:(sampling_rate*40)+2]:
            batch.append(i)
            if len(batch)==batch_size:
                batch_mean = sum(batch)/batch_size
                layer.append(batch_mean)
                batch=[]
        result.append(layer)
    return result

def get_packed_stft(target, batch_size, sampling_rate, i):
    '''
    input:
    target -> sample to apply the tranformation
    batch_size -> lenght of the signal reduction (int)
    sampling_rate -> sampling rate of the signal (int)
    i -> index starting number
    output:
    return pandas series with the label 'stft' 
    starting with index i ready to be concatenated to a pandas dataframe
    '''
    fft_X = {}
    for x in [get_reduced_freq(x, batch_size, sampling_rate) for x in target]:
        fft_X[i] = x
        i+=1
    return pd.Series(fft_X).rename('stft')

def create_df_file_from_patient(patient, file):
    '''
    input:
    patient -> dictionary with patient information extracted from a csv
    file -> path oto store the output dataset
    output:
    void but generated a .pikle file of the pandas dataframe
    '''
    global DONE
    first = True
    record = pd.DataFrame()
    for i, sesion in patient.groupby("seizure_file"):
        seizure_start = list(sesion.seizure_start.values)
        seizure_duration = list(sesion.seizure_duration.values)
        edf_data = mne.io.read_raw_edf(
            drive_path
            + "chb-mit-scalp-eeg-database-1.0.0/"
            + sesion["case"][0]
            + "/"
            + sesion["seizure_file"][0]
            + ".edf/",
            verbose=50,
        )
        if len(diff(channels, edf_data.ch_names)) == 0:
            clear_output(wait=True)
            progress_bar()
            edf_data.drop_channels(diff(edf_data.ch_names, channels))

            # locate and anotate seizures
            seizures = mne.Annotations(
                onset=seizure_start, duration=seizure_duration, description="bad"
            )
            edf_data.set_annotations(seizures)

            # get seizures and split into epochs
            raw_seizures = mne.concatenate_raws(
                edf_data.crop_by_annotations(), verbose=50
            )
            seizures = mne.make_fixed_length_epochs(
                raw_seizures,
                EPOCH_DURATION,
                overlap=OVERLAP_DURATION,
                reject_by_annotation=False,
                verbose=50,
            )

            # get non-seizures and split into epochs
            non_seizures = mne.make_fixed_length_epochs(
                edf_data, EPOCH_DURATION, reject_by_annotation=True, verbose=50
            )

            # computing features for non-seizure
            X = non_seizures._get_data(verbose=50)
            Y = seizures._get_data(verbose=50)
            std_X = np.std(X, axis=2)
            var_X = np.var(X, axis=2)
            max_X = np.max(X, axis=2)
            X_rfft = np.real(rfft(X, axis=2))
            std_X_rfft = np.std(X_rfft, axis=2)
            var_X_rfft = np.var(X_rfft, axis=2)
            max_X_rfft = np.max(X_rfft, axis=2)
            index_X = list(range(len(X)))

            # computing convolutions for non-seizure
            fft_X = get_packed_stft(X, 6, 14, 0)
            
            # computing features for seizure
            std_Y = np.std(Y, axis=2)
            var_Y = np.var(Y, axis=2)
            max_Y = np.max(Y, axis=2)
            Y_rfft = np.real(rfft(Y, axis=2))
            std_Y_rfft = np.std(Y_rfft, axis=2)
            var_Y_rfft = np.var(Y_rfft, axis=2)
            max_Y_rfft = np.max(Y_rfft, axis=2)
            fft_Y = get_packed_stft(Y, 6, 14, len(X))
            index_Y = list(range(len(X), len(X) + len(Y)))

            # computing features for seizure
            df_X = pd.DataFrame(data=std_X, index=index_X, columns=label_std)
            df_X = df_X.join(pd.DataFrame(data=var_X, index=index_X, columns=label_var))
            df_X = df_X.join(pd.DataFrame(data=max_X, index=index_X, columns=label_max))
            df_X = df_X.join(pd.DataFrame(data=std_X_rfft, index=index_X, columns=label_std_rfft))
            df_X = df_X.join(pd.DataFrame(data=var_X_rfft, index=index_X, columns=label_var_rfft))
            df_X = df_X.join(pd.DataFrame(data=max_X_rfft, index=index_X, columns=label_max_rfft))
            df_X = df_X.join(fft_X)

            df_Y = pd.DataFrame(data=std_Y, index=index_Y, columns=label_std)
            df_Y = df_Y.join(pd.DataFrame(data=var_Y, index=index_Y, columns=label_var))
            df_Y = df_Y.join(pd.DataFrame(data=max_Y, index=index_Y, columns=label_max))
            df_Y = df_Y.join(pd.DataFrame(data=std_Y_rfft, index=index_Y, columns=label_std_rfft))
            df_Y = df_Y.join(pd.DataFrame(data=var_Y_rfft, index=index_Y, columns=label_var_rfft))
            df_Y = df_Y.join(pd.DataFrame(data=max_Y_rfft, index=index_Y, columns=label_max_rfft))
            df_Y = df_Y.join(fft_Y)
            df = pd.concat([df_X, df_Y])

            stft = df["stft"]
            df = df.drop(columns=["stft"])
            df =(df - df.mean()) / df.std()
            df.insert(0, "stft", stft, True)
            # merge seizure and non-seizure
            aux = pd.concat(
                [
                    pd.DataFrame(data=0, index=index_X, columns=["seizure"]),
                    pd.DataFrame(data=1, index=index_Y, columns=["seizure"]),
                ]
            )
            df = df.join(aux)
            if record.empty:
                record = df
            else:
                record = pd.concat([record, df])
        else:
            print(diff(channels, edf_data.ch_names))
            print(sesion["seizure_file"], "no channels")
        DONE += 1
    
    # save file
    pickle.dump(df, file)
    file.close()


# loop per patient
for target in train:
    patient = seizure_pointers[seizure_pointers["case"] == target]
    create_df_file_from_patient(
        patient, open("data/train/" + target + ".pickle", "wb")
    )

for target in test:
    patient = seizure_pointers[seizure_pointers["case"] == target]
    create_df_file_from_patient(
        patient, open("data/test/" + target + ".pickle", "wb")
    )
print("compleated")