# data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset

class FRA_Dataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)

        self.X = torch.from_numpy(data["X"]).float()       # (N, L)
        self.P = torch.from_numpy(data["P"]).float()       # optional
        self.meta = torch.from_numpy(data["meta"]).float() # metadata

        self.y_fault = torch.from_numpy(data["y_fault"]).long()
        self.y_cond  = torch.from_numpy(data["y_cond"]).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.X[idx].unsqueeze(0),   # (1, L)
            "meta": self.meta[idx],
            "fault": self.y_fault[idx],
            "cond": self.y_cond[idx]
        }
