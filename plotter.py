import sys
import numpy as np


def get_reduced_freq(target, batch_size, sampling_rate):
    from scipy.fft import rfft
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
    import pandas as pd
    fft_X = {}
    for x in [get_reduced_freq(x, batch_size, sampling_rate) for x in target]:
        fft_X[i] = x
        i+=1
    return pd.Series(fft_X).rename('stft')


def remove_no_neighbor_numbers(numbers):
    result = []
    for i in range(len(numbers)):
        if (i > 0 and i < len(numbers) - 1) or (len(numbers) == 1):
            result.append(numbers[i])
    return result


def _get_model_classes():
    import torch
    import torch.nn as nn

    class FFNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(FFNN, self).__init__()
            self.lin1 = nn.Linear(input_dim, 64)
            self.norm1 = nn.BatchNorm1d(64)
            self.lrelu = nn.LeakyReLU()
            self.drop = nn.Dropout(0.2)

            self.lin2 = nn.Linear(64, 32)
            self.norm2 = nn.BatchNorm1d(32)
            self.lrelu = nn.LeakyReLU()

            self.lin3 = nn.Linear(32, 16)
            self.norm3 = nn.BatchNorm1d(16)
            self.lout = nn.Linear(16, output_dim)

        def forward(self, x):
            y_pred = self.lin1(x)
            y_pred = self.norm1(y_pred)
            y_pred = self.lrelu(y_pred)
            y_pred = self.drop(y_pred)

            y_pred = self.lin2(y_pred)
            y_pred = self.norm2(y_pred)
            y_pred = self.lrelu(y_pred)
            y_pred = self.drop(y_pred)

            y_pred = self.lin3(y_pred)
            y_pred = self.norm3(y_pred)
            y_pred = self.lrelu(y_pred)

            y_pred = self.lout(y_pred)
            return y_pred.squeeze()

    class CNNFFNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(CNNFFNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 3, 3, padding=1, stride=2)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(3, 9, 3, padding=1, stride=2)
            self.lin1 = nn.Linear(input_dim, 512)
            self.norm1 = nn.BatchNorm1d(512)
            self.lrelu = nn.LeakyReLU()
            self.drop = nn.Dropout(0.2)

            self.lin2 = nn.Linear(512, 512)
            self.norm2 = nn.BatchNorm1d(512)
            self.lrelu = nn.LeakyReLU()

            self.lin3 = nn.Linear(512, 128)
            self.norm3 = nn.BatchNorm1d(128)
            self.lout = nn.Linear(128, output_dim)

        def forward(self, time, freq):
            freq = self.conv1(freq)
            freq = self.relu(freq)
            freq = self.drop(freq)

            freq = self.conv2(freq)
            freq = self.relu(freq)

            freq = freq.contiguous().view(freq.shape[0],freq.shape[1]*freq.shape[2]*freq.shape[3])
            features = torch.cat((freq, time), dim=1)

            y_pred = self.lin1(features)
            y_pred = self.norm1(y_pred)
            y_pred = self.lrelu(y_pred)
            y_pred = self.drop(y_pred)

            y_pred = self.lin2(y_pred)
            y_pred = self.norm2(y_pred)
            y_pred = self.lrelu(y_pred)
            y_pred = self.drop(y_pred)

            y_pred = self.lin3(y_pred)
            y_pred = self.norm3(y_pred)
            y_pred = self.lrelu(y_pred)

            y_pred = self.lout(y_pred)
            return y_pred.squeeze()

    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.enc1 = nn.Linear(input_dim, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.enc2 = nn.Linear(64, 32)
            self.bn2 = nn.BatchNorm1d(32)
            self.enc3 = nn.Linear(32, 16)

            self.dec1 = nn.Linear(16, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.dec2 = nn.Linear(32, 64)
            self.bn4 = nn.BatchNorm1d(64)
            self.dec3 = nn.Linear(64, input_dim)

            self.lrelu = nn.LeakyReLU()

        def forward(self, x):
            z = self.lrelu(self.bn1(self.enc1(x)))
            z = self.lrelu(self.bn2(self.enc2(z)))
            z = self.enc3(z)

            out = self.lrelu(self.bn3(self.dec1(z)))
            out = self.lrelu(self.bn4(self.dec2(out)))
            out = self.dec3(out)
            return out

    return FFNN, CNNFFNN, Autoencoder


def process_eeg(option, model_name, threshold):
    """Load EEG data, run inference, return data needed for plotting."""
    import mne
    import torch
    import pickle
    import json
    import pandas as pd
    from scipy.fft import rfft

    FFNN, CNNFFNN, Autoencoder = _get_model_classes()

    drive_path = "raw_data/"

    channels = [
        "P8-O2", "C4-P4", "FP1-F3", "FP2-F8", "CZ-PZ", "FP1-F7",
        "T7-P7", "C3-P3", "FP2-F4", "P4-O2", "F8-T8", "F7-T7",
        "F3-C3", "FZ-CZ", "P3-O1", "P7-O1", "F4-C4",
    ]
    label_std = [n + "-std" for n in channels]
    label_var = [n + "-var" for n in channels]
    label_max = [n + "-max" for n in channels]
    label_std_rfft = [n + "-std_rfft" for n in channels]
    label_var_rfft = [n + "-var_rfft" for n in channels]
    label_max_rfft = [n + "-max_rfft" for n in channels]

    target = option.split('_')[0]
    file = option + ".edf"
    seizure_pointers = pd.read_excel(drive_path + "seizure data.xlsx", index_col=0)
    seizure_pointers["index"] = (
        seizure_pointers["seizure_file"]
        + " "
        + seizure_pointers["seizure_number"].astype(str)
    )

    patient = seizure_pointers[seizure_pointers["index"] == option + " 1"]
    seizure_start = []
    seizure_duration = []
    for _, row in seizure_pointers[seizure_pointers["seizure_file"] == option].iterrows():
        seizure_start.append(row["seizure_start"])
        seizure_duration.append(row["seizure_end"] - row["seizure_start"])

    edf_data = mne.io.read_raw_edf(
        drive_path
        + "chb-mit-scalp-eeg-database-1.0.0/"
        + patient.case.values[0]
        + "/"
        + patient.seizure_file.values[0]
        + ".edf/",
        preload=True,
        verbose=50
    )
    edf_data.drop_channels(list(set(edf_data.ch_names) - set(channels)))

    EPOCH_DURATION = 2
    OVERLAP_DURATION = 1
    samples = mne.make_fixed_length_epochs(
        edf_data, EPOCH_DURATION, overlap=OVERLAP_DURATION,
        reject_by_annotation=False, verbose=50
    )
    samples = samples._get_data(verbose=50)

    std_X = np.std(samples, axis=2)
    var_X = np.var(samples, axis=2)
    max_X = np.max(samples, axis=2)
    X_rfft = np.real(rfft(samples, axis=2))
    std_X_rfft = np.std(X_rfft, axis=2)
    var_X_rfft = np.var(X_rfft, axis=2)
    max_X_rfft = np.max(X_rfft, axis=2)
    fft_X = get_packed_stft(samples, 6, 14, 0)
    index_X = list(range(len(samples)))

    df = pd.DataFrame(data=std_X, index=index_X, columns=label_std)
    df = df.join(pd.DataFrame(data=var_X, index=index_X, columns=label_var))
    df = df.join(pd.DataFrame(data=max_X, index=index_X, columns=label_max))
    df = df.join(pd.DataFrame(data=std_X_rfft, index=index_X, columns=label_std_rfft))
    df = df.join(pd.DataFrame(data=var_X_rfft, index=index_X, columns=label_var_rfft))
    df = df.join(pd.DataFrame(data=max_X_rfft, index=index_X, columns=label_max_rfft))
    df = (df - df.mean()) / df.std()
    df = df.join(fft_X)

    detected = []
    if model_name == "CNN+FFNN":
        model = torch.load("Models/cnnffnn_model.pt")
        model.eval()
        with torch.no_grad():
            for index, item in df.iterrows():
                freq = item.pop("stft")
                freq = torch.tensor([[[x] for x in freq]], requires_grad=False, dtype=torch.float32).permute(0,2,1,3)
                time = torch.tensor(np.array([item.values.astype(np.float32)]), requires_grad=False)
                outputs = model(time, freq)
                predicted = (outputs > float(threshold)).float()
                if predicted == 1:
                    detected.append(index)

    elif model_name == 'FFNN':
        df = df.drop(columns=['stft'])
        model = torch.load("Models/ffnn_model.pt")
        model.eval()
        with torch.no_grad():
            for index, item in df.iterrows():
                time = torch.tensor(np.array([item.values.astype(np.float32)]), requires_grad=False)
                outputs = model(time)
                predicted = (outputs > float(threshold)).float()
                if predicted == 1:
                    detected.append(index)

    elif model_name == 'SVM':
        model = pickle.load(open('Models/svm_model.pt', 'rb'))
        df = df.drop(columns=['stft'])
        for index, item in df.iterrows():
            predicted = model.predict(item.values.reshape(1, -1))[0]
            if predicted == 1:
                detected.append(index)

    elif model_name == 'RF':
        model = pickle.load(open('Models/rf_model.pt', 'rb'))
        df = df.drop(columns=['stft'])
        for index, item in df.iterrows():
            predicted = model.predict(item.values.reshape(1, -1))[0]
            if predicted == 1:
                detected.append(index)

    elif model_name == 'KNN':
        model = pickle.load(open('Models/knn_model.pt', 'rb'))
        df = df.drop(columns=['stft'])
        for index, item in df.iterrows():
            predicted = model.predict(item.values.reshape(1, -1))[0]
            if predicted == 1:
                detected.append(index)

    elif model_name == 'Autoencoder':
        df = df.drop(columns=['stft'])
        ae_model = Autoencoder(input_dim=102)
        ae_model.load_state_dict(torch.load("Models/autoencoder_model.pt", weights_only=True))
        ae_model.eval()
        with open('Models/autoencoder_threshold.json') as f:
            ae_config = json.load(f)
        ae_thresh = ae_config['threshold']
        with torch.no_grad():
            features = torch.tensor(df.values.astype(np.float32))
            recon = ae_model(features)
            mae = torch.mean(torch.abs(recon - features), dim=1)
            for i in range(len(mae)):
                if mae[i].item() > ae_thresh:
                    detected.append(i)

    detected = remove_no_neighbor_numbers(detected)

    sfreq = edf_data.info['sfreq']
    ch_names = edf_data.ch_names
    raw_data = edf_data.get_data()

    return {
        'raw_data': raw_data,
        'sfreq': sfreq,
        'ch_names': ch_names,
        'seizure_start': seizure_start,
        'seizure_duration': seizure_duration,
        'detected': detected,
    }


if __name__ == "__main__":
    option = sys.argv[1]
    model = sys.argv[2]
    threshold = sys.argv[3]
    result = process_eeg(option, model, threshold)
    print(f"Detected {len(result['detected'])} seizure epochs")
