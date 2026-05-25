import numpy as np
import pandas as pd
import mne, os

EPOCH_DURATION = 6
OVERLAP_DURATION = 3

train = [
    "chb01", "chb02", "chb03", "chb04", "chb05", "chb06",
    "chb10", "chb11", "chb14", "chb15", "chb16", "chb17",
    "chb18", "chb19", "chb21", "chb22", "chb23", "chb24",
]

test = ["chb07", "chb08", "chb09"]

val = ["chb13", "chb20"]

drive_path = "raw_data/"
seizure_pointers = pd.read_excel(drive_path + "seizure data.xlsx", index_col=0)
seizure_pointers["index"] = (
    seizure_pointers["seizure_file"]
    + " "
    + seizure_pointers["seizure_number"].astype(str)
)
seizure_pointers = seizure_pointers.set_index("index")

channels = [
    "P8-O2", "C4-P4", "FP1-F3", "FP2-F8", "CZ-PZ", "FP1-F7", "T7-P7",
    "C3-P3", "FP2-F4", "P4-O2", "F8-T8", "F7-T7", "F3-C3", "FZ-CZ",
    "P3-O1", "P7-O1", "F4-C4",
]

os.makedirs("data_v3/train", exist_ok=True)
os.makedirs("data_v3/valid", exist_ok=True)
os.makedirs("data_v3/test", exist_ok=True)

DONE = 1
TOTAL = len(seizure_pointers[seizure_pointers["case"].isin(train + test + val)]["seizure_file"].unique())

def progress_bar():
    percent = 100 * (DONE / float(TOTAL))
    bar = "█" * int(percent) + "-" * (100 - int(percent))
    print(f"\r|{bar}| {percent: .2f}%", end="\r")


def diff(lst1, lst2):
    return list(set(lst1) - set(lst2))


def process_patient(patient, output_path):
    global DONE
    all_epochs = []
    all_labels = []

    for _, session in patient.groupby("seizure_file"):
        DONE += 1
        progress_bar()
        seizure_start = list(session.seizure_start.values)
        seizure_duration = list(session.seizure_duration.values)

        edf_path = (
            drive_path
            + "chb-mit-scalp-eeg-database-1.0.0/"
            + session["case"].iloc[0]
            + "/"
            + session["seizure_file"].iloc[0]
            + ".edf/"
        )

        edf_data = mne.io.read_raw_edf(edf_path, verbose=50)

        if len(diff(channels, edf_data.ch_names)) != 0:
            print(f"  Skipping {session['seizure_file'].iloc[0]}: missing channels")
            continue

        edf_data.drop_channels(diff(edf_data.ch_names, channels))

        seizures = mne.Annotations(
            onset=seizure_start, duration=seizure_duration, description="bad"
        )
        edf_data.set_annotations(seizures)

        # Seizure epochs (with overlap)
        raw_seizures = mne.concatenate_raws(
            edf_data.crop_by_annotations(), verbose=50
        )
        seizure_epochs = mne.make_fixed_length_epochs(
            raw_seizures, EPOCH_DURATION,
            overlap=OVERLAP_DURATION,
            reject_by_annotation=False, verbose=50,
        )
        Y = seizure_epochs._get_data(verbose=50)  # (n_seizure, 17, 512)

        # Non-seizure epochs (no overlap, reject seizure periods)
        non_seizure_epochs = mne.make_fixed_length_epochs(
            edf_data, EPOCH_DURATION,
            reject_by_annotation=True, verbose=50,
        )
        X = non_seizure_epochs._get_data(verbose=50)  # (n_normal, 17, 512)

        # Z-score normalize per channel across the recording
        combined = np.concatenate([X, Y], axis=0)  # (n_total, 17, 512)
        for ch in range(combined.shape[1]):
            ch_mean = combined[:, ch, :].mean()
            ch_std = combined[:, ch, :].std()
            if ch_std > 0:
                combined[:, ch, :] = (combined[:, ch, :] - ch_mean) / ch_std

        labels = np.concatenate([
            np.zeros(X.shape[0], dtype=np.int32),
            np.ones(Y.shape[0], dtype=np.int32),
        ])

        all_epochs.append(combined)
        all_labels.append(labels)

    if all_epochs:
        epochs_arr = np.concatenate(all_epochs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        np.savez_compressed(output_path, epochs=epochs_arr, labels=labels_arr)
        n_normal = (labels_arr == 0).sum()
        n_seizure = (labels_arr == 1).sum()
        print(f"  Saved {output_path}: {epochs_arr.shape[0]} epochs "
              f"({n_normal} normal, {n_seizure} seizure), shape per epoch: {epochs_arr.shape[1:]}")
    else:
        print(f"  No valid sessions for {output_path}")


print("Processing training patients...")
for target in train:
    print(f"Patient: {target}")
    patient = seizure_pointers[seizure_pointers["case"] == target]
    process_patient(patient, f"data_v3/train/{target}.npz")

print("\nProcessing test patients...")
for target in test:
    print(f"Patient: {target}")
    patient = seizure_pointers[seizure_pointers["case"] == target]
    process_patient(patient, f"data_v3/test/{target}.npz")

print("\nProcessing validation patients...")
for target in val:
    print(f"Patient: {target}")
    patient = seizure_pointers[seizure_pointers["case"] == target]
    process_patient(patient, f"data_v3/valid/{target}.npz")

print("\nDone!")
