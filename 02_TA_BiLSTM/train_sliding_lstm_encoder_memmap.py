import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

DATA_DIR = os.path.join(BASE, "data", "processed", "sliding_memmap")
MODEL_DIR = os.path.join(BASE, "data", "processed", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "sliding_lstm_encoder.pt")

# ==========================================================
# DATASET
# ==========================================================

class SlidingDataset(Dataset):

    def __init__(self, prefix):

        self.X = np.load(
            os.path.join(DATA_DIR, f"{prefix}_XM.npy"),
            mmap_mode="r"
        )

        self.y = np.load(
            os.path.join(DATA_DIR, f"{prefix}_y.npy"),
            mmap_mode="r"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return x, y


# ==========================================================
# ATTENTION
# ==========================================================

class TemporalAttention(nn.Module):

    def __init__(self, hidden):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        w = torch.softmax(self.score(x), dim=1)
        context = torch.sum(w * x, dim=1)

        return context


# ==========================================================
# MODEL
# ==========================================================

class SlidingEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm = nn.LayerNorm(80)

        self.lstm = nn.LSTM(
            input_size=80,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )

        self.attn = TemporalAttention(64)

        self.embed = nn.Linear(128, 64)

        self.drop = nn.Dropout(0.3)

        self.cls = nn.Linear(64, 1)

    def forward(self, x):

        x = self.norm(x)

        out, _ = self.lstm(x)

        context = self.attn(out)

        z = torch.relu(self.embed(context))

        z = self.drop(z)

        logit = self.cls(z)

        return logit.squeeze(), z


# ==========================================================
# TRAINING
# ==========================================================

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("SLIDING LSTM ENCODER (MEMMAP)")
    print("=" * 60)
    print("Device :", device)

    if torch.cuda.is_available():
        print("GPU    :", torch.cuda.get_device_name(0))

    train_ds = SlidingDataset("train_sliding")
    valid_ds = SlidingDataset("validation_sliding")

    print("Train windows :", len(train_ds))
    print("Valid windows :", len(valid_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = SlidingEncoder().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        factor=0.5
    )

    scaler = torch.amp.GradScaler("cuda")

    EPOCHS = 8
    PATIENCE = 3

    best_loss = float("inf")
    wait = 0

    print("\nTraining Started\n")

    for epoch in range(EPOCHS):

        # ---------------- TRAIN ----------------

        model.train()
        train_loss = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            ncols=110
        )

        for X, y in pbar:

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                logits, _ = model(X)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        # ---------------- VALID ----------------

        model.eval()
        val_loss = 0

        with torch.no_grad():

            for X, y in valid_loader:

                X = X.to(device)
                y = y.to(device)

                with torch.amp.autocast("cuda"):

                    logits, _ = model(X)
                    loss = criterion(logits, y)

                val_loss += loss.item()

        val_loss /= len(valid_loader)

        scheduler.step(val_loss)

        print(
            f"\nEpoch {epoch+1} Completed | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f}\n"
        )

        if val_loss < best_loss:

            best_loss = val_loss
            wait = 0

            torch.save(model.state_dict(), MODEL_PATH)

        else:

            wait += 1

        if wait >= PATIENCE:

            print("Early Stopping Triggered")
            break

    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("Best Validation Loss :", round(best_loss, 4))
    print("Model Saved :", MODEL_PATH)


if __name__ == "__main__":
    train()