"""Test script to verify the autoencoder code works end-to-end."""
import pandas as pd
import pickle, torch, glob, copy
import numpy as np
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- Load balanced data (for test set and metrics) ---
with open('data/test.pickle', 'rb') as f:
    test = pickle.load(f)
with open('data/train.pickle', 'rb') as f:
    train = pickle.load(f)

train = train.dropna()
train = train.reset_index(drop=['index'])
y_train = train['seizure']
X_train = train.drop(columns='seizure')

test = test.dropna()
test = test.reset_index(drop=['index'])
y_test = test['seizure']
X_test = test.drop(columns='seizure')

X_train1 = X_train.drop(columns=["stft"])
X_test1 = X_test.drop(columns=["stft"])

def metrics(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    d = {
        "metric": ['sensitivity', 'specificity', 'accuracy'],
        'precision': [
            f'{(tp/(tp+fn))*100:.3f}%',
            f'{(tn/(fp+tn))*100:.3f}%',
            f'{((tp+tn)/(tn+fp+fn+tp))*100:.3f}%'
        ]
    }
    return pd.DataFrame.from_dict(d)

# --- Load all non-seizure training data ---
train_pickles = sorted(glob.glob('data/train/*.pickle'))
normal_frames = []
for path in train_pickles:
    with open(path, 'rb') as f:
        patient_df = pickle.load(f)
    patient_normal = patient_df[patient_df['seizure'] == 0].drop(columns=['stft', 'seizure'])
    normal_frames.append(patient_normal)

X_train_normal = pd.concat(normal_frames, ignore_index=True).dropna()
print(f"Non-seizure training samples (all patients): {X_train_normal.shape[0]}")
print(f"Features: {X_train_normal.shape[1]}")
print(f"Balanced train.pickle non-seizure count:     {(y_train == 0).sum()}")

# --- Model definition ---
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

train_tensor = torch.tensor(X_train_normal.values.astype(np.float32))
ae_dataset = torch.utils.data.TensorDataset(train_tensor)
ae_loader = torch.utils.data.DataLoader(ae_dataset, batch_size=64, shuffle=True, drop_last=True)

autoencoder = Autoencoder(input_dim=X_train_normal.shape[1])
ae_criterion = nn.MSELoss()
ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
print(autoencoder)

# --- Training ---
num_epochs = 50
ae_train_loss = []
best_ae = None
best_ae_loss = float('inf')

autoencoder.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for (batch,) in ae_loader:
        reconstruction = autoencoder(batch)
        loss = ae_criterion(reconstruction, batch)
        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(ae_loader)
    ae_train_loss.append(avg_loss)

    if avg_loss < best_ae_loss:
        best_ae_loss = avg_loss
        best_ae = copy.deepcopy(autoencoder.state_dict())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

autoencoder.load_state_dict(best_ae)

# --- Compute threshold ---
autoencoder.eval()
with torch.no_grad():
    reconstructions = autoencoder(train_tensor)
    mae_per_sample = torch.mean(torch.abs(reconstructions - train_tensor), dim=1)

ae_threshold = mae_per_sample.mean() + mae_per_sample.std()
print(f"\nMean MAE (normal training data): {mae_per_sample.mean():.6f}")
print(f"Std MAE (normal training data):  {mae_per_sample.std():.6f}")
print(f"Threshold (mean + 1*std):        {ae_threshold:.6f}")

# --- Evaluate on test set ---
test_tensor = torch.tensor(X_test1.values.astype(np.float32))

autoencoder.eval()
with torch.no_grad():
    test_recon = autoencoder(test_tensor)
    test_mae = torch.mean(torch.abs(test_recon - test_tensor), dim=1)

y_pred_ae = (test_mae > ae_threshold).float().numpy()
print(f"\n--- Autoencoder Evaluation ---")
print(metrics(y_test.values, y_pred_ae).to_string(index=False))

print("\nTest passed!")
