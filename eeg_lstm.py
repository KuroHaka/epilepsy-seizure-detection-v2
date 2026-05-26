import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

def convert_npz_to_mmap(npz_paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for path in npz_paths:
        stem = Path(path).stem
        out_epochs = out_dir / f"{stem}_epochs.npy"
        out_labels = out_dir / f"{stem}_labels.npy"
        if out_epochs.exists():
            continue
        print(f"Converting {stem}...")
        data = np.load(path)
        np.save(out_epochs, data["epochs"].astype(np.float32))
        np.save(out_labels, data["labels"].astype(np.int64))

class EEGDataset(Dataset):
    def __init__(self, mmap_dir):
        epoch_files = sorted(Path(mmap_dir).glob("*_epochs.npy"))
        label_files = sorted(Path(mmap_dir).glob("*_labels.npy"))

        # Memory-map all files — loads instantly, zero RAM used
        self.epochs = [np.load(f, mmap_mode="r") for f in epoch_files]
        self.labels = [np.load(f, mmap_mode="r") for f in label_files]

        # Build index from shape only — never touches file contents
        self.index_map = np.array([
            (file_idx, epoch_idx)
            for file_idx, arr in enumerate(self.epochs)
            for epoch_idx in range(arr.shape[0])  # .shape is metadata, not data
        ], dtype=np.int32)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, epoch_idx = self.index_map[idx]
        return (
            self.epochs[file_idx][epoch_idx].copy(),  # .copy() needed for mmap
            self.labels[file_idx][epoch_idx].copy(),
        )

class EEGLSTM(nn.Module):
    """
    Bidirectional LSTM for EEG seizure detection.

    Args:
        n_channels:  EEG channels as input features (17)
        hidden_size: LSTM hidden units
        n_layers:    stacked LSTM layers
        dropout:     dropout between LSTM layers
        n_classes:   output classes (2)
        bidirectional: attend to past and future context
    """
    def __init__(
        self,
        n_channels    = 17,
        hidden_size   = 128,
        n_layers      = 3,
        dropout       = 0.3,
        n_classes     = 2,
        bidirectional = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size   = hidden_size
        self.n_layers      = n_layers
        d = hidden_size * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size    = n_channels,
            hidden_size   = hidden_size,
            num_layers    = n_layers,
            batch_first   = True,
            dropout        = dropout if n_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, n_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, channels/features, time points) -> (batch, time points, channels/features)
        out, _ = self.lstm(x) 
        x = out.mean(dim=1)
        return self.classifier(x)

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for epochs, labels in loader:
        epochs, labels = epochs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(epochs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for epochs, labels in loader:
        epochs, labels = epochs.to(device), labels.to(device)
        logits = model(epochs)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        probs       = torch.softmax(logits, dim=-1)[:, 1]
        preds       = (probs > threshold).long()
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().float()

    precision   = tp / (tp + fp + 1e-8)
    recall      = tp / (tp + fn + 1e-8)
    f1          = 2 * precision * recall / (precision + recall + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "loss":        total_loss / len(loader),
        "accuracy":    correct / total,
        "precision":   precision.item(),
        "recall":      recall.item(),
        "f1":          f1.item(),
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
    }