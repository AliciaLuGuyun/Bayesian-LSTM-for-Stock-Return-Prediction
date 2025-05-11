# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LSTMDataset(Dataset):
    def __init__(self, series, window_size):
        self.series = torch.tensor(series, dtype=torch.float32).unsqueeze(-1)  # shape: (T, 1)
        self.window_size = window_size

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.window_size]
        y = self.series[idx + self.window_size]
        return x, y


def create_lstm_dataloaders(train_series, test_series, window_size=20, batch_size=32):
    # Create dataset
    train_dataset = LSTMDataset(train_series, window_size)

    # DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare test data (not using DataLoader here)
    test_seq = []
    test_target = []
    series_tensor = torch.tensor(test_series, dtype=torch.float32).unsqueeze(-1)
    for i in range(len(series_tensor) - window_size):
        test_seq.append(series_tensor[i : i + window_size])
        test_target.append(series_tensor[i + window_size])
    test_seq = torch.stack(test_seq)
    test_target = torch.stack(test_target)

    return train_loader, test_seq, test_target