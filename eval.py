# eval.py
import torch
from torch.utils.data import DataLoader     # <-- FIXED
from data_loader import FRA_Dataset
from model import HybridFRA
from sklearn.metrics import classification_report

TEST = "./processed_fra_pytorch/test.npz"

def evaluate():
    ds = FRA_Dataset(TEST)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HybridFRA(
        seq_len=ds.X.shape[1],
        meta_dim=ds.meta.shape[1],
        num_fault_classes=len(ds.y_fault.unique())
    )
    model.load_state_dict(torch.load("fra_model.pth", map_location=device))
    model.to(device)
    model.eval()

    loader = DataLoader(ds, batch_size=16)
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            meta = batch["meta"].to(device)
            y = batch["fault"].cpu().numpy()

            logits = model(x, meta)
            p = logits.argmax(1).cpu().numpy()

            preds.extend(p)
            trues.extend(y)

    print(classification_report(trues, preds))

if __name__ == "__main__":
    evaluate()
