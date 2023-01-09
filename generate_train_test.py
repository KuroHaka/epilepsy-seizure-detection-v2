import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
import csv, mne, math
from IPython.display import clear_output

# constants
EPOCH_DURATION = 2
OVERLAP_DURATION = 1

train = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb10",
    "chb11",
    "chb12",
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
]

valid = ["chb24","chb09"]
test = ["chb07", "chb08"]

drive_path = "C:/Users/Eugene Chen/Desktop/UNI/Project/Data/"
seizure_pointers = pd.read_excel(drive_path + "seizure data.xlsx", index_col=0)
seizure_pointers["index"] = (
    seizure_pointers["seizure_file"]
    + " "
    + seizure_pointers["seizure_number"].astype(str)
)
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

DONE = 1
TOTAL = 0
for t in train + test:
    TOTAL += seizure_pointers[seizure_pointers.case == t].shape[0]

label_std = [n + "-std" for n in channels]
label_var = [n + "-var" for n in channels]
label_max = [n + "-max" for n in channels]
label_skw = [n + "-skw" for n in channels]
label_krt = [n + "-krt" for n in channels]
label_0_5_fft = [n + "-0_5_fft" for n in channels]
label_6_15_fft = [n + "-6_15_fft" for n in channels]
label_16_40_fft = [n + "-16_40_fft" for n in channels]
label_41_80_fft = [n + "-41_80_fft" for n in channels]


def progress_bar():
    percent = 100 * (DONE / float(TOTAL))
    bar = "█" * int(percent) + "-" * (100 - int(percent))
    print(f"\r|{bar}| {percent: .2f}%", end="\r")


def diff(lst1, lst2):
    return list(set(lst1) - set(lst2))


def create_csv_file_from_patient(patient, file):
    global DONE
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
            seizures = mne.Annotations(
                onset=seizure_start, duration=seizure_duration, description="bad"
            )
            edf_data.set_annotations(seizures)
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
            non_seizures = mne.make_fixed_length_epochs(
                edf_data, EPOCH_DURATION, reject_by_annotation=True, verbose=50
            )
            X = non_seizures._get_data(verbose=50)
            Y = seizures._get_data(verbose=50)
            std_X = np.std(X, axis=2)
            var_X = np.var(X, axis=2)
            max_X = np.max(X, axis=2)
            skw_X = skew(X, axis=2)
            krt_X = kurtosis(X, axis=2)
            X_rfft = np.real(rfft(X, axis=2))
            std_X_rfft = np.std(X_rfft, axis=2)
            var_X_rfft = np.var(X_rfft, axis=2)
            max_X_rfft = np.max(X_rfft, axis=2)
            skw_X_rfft = skew(X_rfft, axis=2)
            krt_X_rfft = kurtosis(X_rfft, axis=2)
            index_X = list(range(len(X)))

            std_Y = np.std(Y, axis=2)
            var_Y = np.var(Y, axis=2)
            max_Y = np.max(Y, axis=2)
            skw_Y = skew(Y, axis=2)
            krt_Y = kurtosis(Y, axis=2)
            Y_rfft = np.real(rfft(Y, axis=2))
            std_Y_rfft = np.std(Y_rfft, axis=2)
            var_Y_rfft = np.var(Y_rfft, axis=2)
            max_Y_rfft = np.max(Y_rfft, axis=2)
            skw_Y_rfft = skew(Y_rfft, axis=2)
            krt_Y_rfft = kurtosis(Y_rfft, axis=2)
            index_Y = list(range(len(X), len(X) + len(Y)))

            df_X = pd.DataFrame(data=std_X, index=index_X, columns=label_std)
            df_X = df_X.join(pd.DataFrame(data=var_X, index=index_X, columns=label_var))
            df_X = df_X.join(pd.DataFrame(data=max_X, index=index_X, columns=label_max))
            df_X = df_X.join(pd.DataFrame(data=skw_X, index=index_X, columns=label_skw))
            df_X = df_X.join(pd.DataFrame(data=krt_X, index=index_X, columns=label_krt))
            df_X = df_X.join(
                pd.DataFrame(data=var_X_rfft, index=index_X, columns=label_var_rfft)
            )
            df_X = df_X.join(
                pd.DataFrame(data=max_X_rfft, index=index_X, columns=label_max_rfft)
            )
            df_X = df_X.join(
                pd.DataFrame(data=skw_X_rfft, index=index_X, columns=label_skw_rfft)
            )
            df_X = df_X.join(
                pd.DataFrame(data=krt_X_rfft, index=index_X, columns=label_krt_rfft)
            )

            df_Y = pd.DataFrame(data=std_Y, index=index_Y, columns=label_std)
            df_Y = df_Y.join(pd.DataFrame(data=var_Y, index=index_Y, columns=label_var))
            df_Y = df_Y.join(pd.DataFrame(data=max_Y, index=index_Y, columns=label_max))
            df_Y = df_Y.join(pd.DataFrame(data=skw_Y, index=index_Y, columns=label_skw))
            df_Y = df_Y.join(pd.DataFrame(data=krt_Y, index=index_Y, columns=label_krt))
            df_Y = df_Y.join(
                pd.DataFrame(data=var_Y_rfft, index=index_Y, columns=label_var_rfft)
            )
            df_Y = df_Y.join(
                pd.DataFrame(data=max_Y_rfft, index=index_Y, columns=label_max_rfft)
            )
            df_Y = df_Y.join(
                pd.DataFrame(data=skw_Y_rfft, index=index_Y, columns=label_skw_rfft)
            )
            df_Y = df_Y.join(
                pd.DataFrame(data=krt_Y_rfft, index=index_Y, columns=label_krt_rfft)
            )

            df = pd.concat([df_X, df_Y])
            df = (df - df.mean()) / df.std()
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
            print(sesion["seizure_file"], "no channels")
        DONE += 1
    df.to_csv(file, index=False)
    file.close()


# loop per patient
for target in train:
    patient = seizure_pointers[seizure_pointers["case"] == target]
    create_csv_file_from_patient(
        patient, open("Code/data/train/" + target + ".csv", "w")
    )

for target in test:
    patient = seizure_pointers[seizure_pointers["case"] == target]
    create_csv_file_from_patient(
        patient, open("Code/data/test/" + target + ".csv", "w")
    )
