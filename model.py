# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=9, padding=4)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))

class HybridFRA(nn.Module):
    def __init__(self, seq_len=2048, meta_dim=7, num_fault_classes=6):
        super().__init__()

        # CNN branch
        self.c1 = ConvBlock(1, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)

        final_len = seq_len // (2**4)
        self.flat_dim = 128 * final_len

        # Metadata branch
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_fault_classes)
        )

    def forward(self, x, meta):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)

        x = x.flatten(1)
        m = self.meta(meta)
        out = torch.cat([x, m], dim=1)

        return self.fc(out)
