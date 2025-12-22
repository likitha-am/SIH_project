# infer.py
import torch
import numpy as np
from model import HybridFRA
from data_loader import FRA_Dataset

def infer(idx=0):
    ds = FRA_Dataset("./processed_fra_pytorch/test.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HybridFRA(seq_len=ds.X.shape[1], meta_dim=ds.meta.shape[1], num_fault_classes=len(ds.y_fault.unique()))
    model.load_state_dict(torch.load("fra_model.pth", map_location=device))
    model.to(device)
    model.eval()

    x = ds.X[idx].unsqueeze(0).unsqueeze(0).to(device)
    meta = ds.meta[idx].unsqueeze(0).to(device)

    logits = model(x, meta)
    pred_class = logits.argmax(1).item()

    print("Predicted Fault Class:", pred_class)

if __name__ == "__main__":
    infer(0)
