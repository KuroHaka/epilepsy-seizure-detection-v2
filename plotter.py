import mne, torch, pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from scipy.fft import rfft



class FFNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FFNN, self).__init__()
        # define layers
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
        # define layers
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 9, 3, padding=1, stride=2)
        self.lin1 = nn.Linear(input_dim, 512)
        #normalize the output data from each of the 50 batch
        self.norm1 = nn.BatchNorm1d(512)
        self.lrelu = nn.LeakyReLU()
        # 20% dropout to avoid overfitting
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


class Plotter:

    @staticmethod
    def get_reduced_freq(target, batch_size, sampling_rate):
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

    @staticmethod
    def get_packed_stft(target, batch_size, sampling_rate, i):
        fft_X = {}
        for x in [Plotter.get_reduced_freq(x, batch_size, sampling_rate) for x in target]:
            fft_X[i] = x
            i+=1
        return pd.Series(fft_X).rename('stft')

    @staticmethod
    def remove_no_neighbor_numbers(numbers):
        result = []
        for i in range(len(numbers)):
            if (i > 0 and i < len(numbers) - 1) or (len(numbers) == 1):
                result.append(numbers[i])
        return result

    @staticmethod
    def plot_results(option, model, threshold):
        drive_path = "raw_data/"
        seizure_pointers = pd.read_excel(drive_path + "seizure data.xlsx", index_col=0)
        seizure_pointers["index"] = (
            seizure_pointers["seizure_file"]
            + " "
            + seizure_pointers["seizure_number"].astype(str)
        )

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
        label_std = [n + "-std" for n in channels]
        label_var = [n + "-var" for n in channels]
        label_max = [n + "-max" for n in channels]
        label_std_rfft = [n + "-std_rfft" for n in channels]
        label_var_rfft = [n + "-var_rfft" for n in channels]
        label_max_rfft = [n + "-max_rfft" for n in channels]
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
        seizures = mne.Annotations(
                        onset=seizure_start, duration=seizure_duration, description="ground true"
                    )
        edf_data.set_annotations(seizures)
        EPOCH_DURATION = 2
        OVERLAP_DURATION = 1
        samples = mne.make_fixed_length_epochs(edf_data, EPOCH_DURATION, overlap=OVERLAP_DURATION, reject_by_annotation=False, verbose=50)
        samples = samples._get_data(verbose=50)
        edf_data.plot(color='green', duration=50, start=seizure_start[0]-1, scalings=dict(mag=1e-12, grad=577e-11, eeg=250e-6, eog=150e-6, ecg=5e-4,
                    emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
                    resp=1, chpi=1e-4, whitened=1e2))
        std_X = np.std(samples, axis=2)
        var_X = np.var(samples, axis=2)
        max_X = np.max(samples, axis=2)
        X_rfft = np.real(rfft(samples, axis=2))
        std_X_rfft = np.std(X_rfft, axis=2)
        var_X_rfft = np.var(X_rfft, axis=2)
        max_X_rfft = np.max(X_rfft, axis=2)
        fft_X = Plotter.get_packed_stft(samples, 6, 14, 0)
        index_X = list(range(len(samples)))
        df = pd.DataFrame(data=std_X, index=index_X, columns=label_std)
        df = df.join(pd.DataFrame(data=var_X, index=index_X, columns=label_var))
        df = df.join(pd.DataFrame(data=max_X, index=index_X, columns=label_max))
        df = df.join(pd.DataFrame(data=std_X_rfft, index=index_X, columns=label_std_rfft))
        df = df.join(pd.DataFrame(data=var_X_rfft, index=index_X, columns=label_var_rfft))
        df = df.join(pd.DataFrame(data=max_X_rfft, index=index_X, columns=label_max_rfft))
        df =(df - df.mean()) / df.std()
        df = df.join(fft_X)

        detected = []
        if model == "CNN+FFNN":
            model = torch.load("Models/cnnffnn_model.pt")
            model.eval()
            with torch.no_grad():
                for index, item in df.iterrows():
                    freq = item.pop("stft")
                    freq = torch.tensor([[[x] for x in freq]], requires_grad=False, dtype=torch.float32).permute(0,2,1,3)
                    time = torch.tensor(np.array([item.values.astype(np.float32)]), requires_grad=False)
                    outputs = model(time, freq)
                    predicted = (outputs>float(threshold)).float()
                    if predicted == 1:
                        detected.append(index)

        elif model == 'FFNN':
            df = df.drop(columns=['stft'])
            model = torch.load("Models/ffnn_model.pt")
            model.eval()
            with torch.no_grad():
                for index, item in df.iterrows():
                    time = torch.tensor(np.array([item.values.astype(np.float32)]), requires_grad=False)
                    outputs = model(time)
                    predicted = (outputs>float(threshold)).float()
                    if predicted == 1:
                        detected.append(index)

        elif model == 'SVM':
            model = pickle.load(open('Models/svm_model.pt', 'rb'))
            df = df.drop(columns=['stft'])
            for index, item in df.iterrows():
                predicted = model.predict(item.values.reshape(1,-1))[0]
                if predicted == 1:
                        detected.append(index)

        elif model == 'RF':
            model = pickle.load(open('Models/rf_model.pt', 'rb'))
            df = df.drop(columns=['stft'])
            for index, item in df.iterrows():
                predicted = model.predict(item.values.reshape(1,-1))[0]
                if predicted == 1:
                        detected.append(index)

        elif model == 'KNN':
            model = pickle.load(open('Models/knn_model.pt', 'rb'))
            df = df.drop(columns=['stft'])
            for index, item in df.iterrows():
                predicted = model.predict(item.values.reshape(1,-1))[0]
                if predicted == 1:
                        detected.append(index)

        detected = Plotter.remove_no_neighbor_numbers(detected)
        seizures_pred = mne.Annotations(
                        onset=detected, duration=[1]*len(detected), description="predicted"
                    )
        edf_data.set_annotations(seizures_pred)
        edf_data.plot(color='red', duration=50, start=seizure_start[0], scalings=dict(mag=1e-12, grad=577e-11, eeg=250e-6, eog=150e-6, ecg=5e-4,
            emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
            resp=1, chpi=1e-4, whitened=1e2))
        plt.show()
