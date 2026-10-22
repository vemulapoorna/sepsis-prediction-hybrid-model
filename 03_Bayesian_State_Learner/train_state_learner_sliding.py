import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

DBN_DIR = os.path.join(BASE, "data", "processed", "dbn")
MODEL_DIR = os.path.join(BASE, "data", "processed", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# DATASET
# ==========================================================

class SlidingDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        feature_cols = [f"z{i}" for i in range(1, 65)]

        self.X = self.df[feature_cols].values.astype(np.float32)
        self.y = self.df["SepsisLabel"].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx])
        )


# ==========================================================
# FOCAL LOSS
# ==========================================================

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.bce = nn.BCELoss(reduction="none")

    def forward(self, pred, target):

        bce = self.bce(pred, target)

        pt = torch.exp(-bce)

        loss = self.alpha * ((1 - pt) ** self.gamma) * bce

        return loss.mean()


# ==========================================================
# MODEL
# ==========================================================

class BayesianStateLearner(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20)

        )

        self.state_layer = nn.Linear(64, 4)

        self.register_buffer(
            "risk_levels",
            torch.tensor(
                [0.05, 0.30, 0.60, 0.90],
                dtype=torch.float32
            )
        )

    def forward(self, x):

        feat = self.encoder(x)

        logits = self.state_layer(feat)

        temperature = 0.7

        state_prob = torch.softmax(
            logits / temperature,
            dim=1
        )

        sepsis_prob = torch.sum(
            state_prob * self.risk_levels,
            dim=1
        )

        return sepsis_prob, state_prob


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("IMPROVED SUPERVISED BAYESIAN STATE LEARNER")
    print("=" * 60)
    print("Device :", DEVICE)

    if torch.cuda.is_available():
        print("GPU    :", torch.cuda.get_device_name(0))

    # ------------------------------------------------------

    train_ds = SlidingDataset(
        os.path.join(DBN_DIR, "train_sliding_temporal_embeddings.csv")
    )

    val_ds = SlidingDataset(
        os.path.join(DBN_DIR, "validation_sliding_temporal_embeddings.csv")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1024,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("Train Windows :", len(train_ds))
    print("Valid Windows :", len(val_ds))

    # ------------------------------------------------------

    model = BayesianStateLearner().to(DEVICE)

    criterion = FocalLoss(alpha=0.75, gamma=2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_loss = 999
    patience = 8
    wait = 0

    print("\nTraining Started\n")

    for epoch in range(30):

        # ---------------- TRAIN ----------------

        model.train()

        train_loss = 0

        for X, y in train_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            pred, _ = model(X)

            loss = criterion(pred, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------- VALID ----------------

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for X, y in val_loader:

                X = X.to(DEVICE)
                y = y.to(DEVICE)

                pred, _ = model(X)

                loss = criterion(pred, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0

        else:

            wait += 1

            if wait >= patience:

                print("\nEarly Stopping Triggered")
                break

    # ------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------

    torch.save(
        best_weights,
        os.path.join(MODEL_DIR, "state_learner_sliding.pt")
    )

    model.load_state_dict(best_weights)

    print("\nBest Validation Loss :", round(best_loss, 4))

    # ------------------------------------------------------
    # EXPORT STATES
    # ------------------------------------------------------

    def export_states(input_csv, output_csv):

        df = pd.read_csv(os.path.join(DBN_DIR, input_csv))

        feature_cols = [f"z{i}" for i in range(1, 65)]

        X = torch.tensor(
            df[feature_cols].values.astype(np.float32)
        ).to(DEVICE)

        model.eval()

        with torch.no_grad():

            sepsis_prob, state_prob = model(X)

        state_prob = state_prob.cpu().numpy()
        sepsis_prob = sepsis_prob.cpu().numpy()

        df["State"] = np.argmax(state_prob, axis=1)

        df["P_S0"] = state_prob[:, 0]
        df["P_S1"] = state_prob[:, 1]
        df["P_S2"] = state_prob[:, 2]
        df["P_S3"] = state_prob[:, 3]

        df["PredictedProbability"] = sepsis_prob

        df.to_csv(
            os.path.join(DBN_DIR, output_csv),
            index=False
        )

    export_states(
        "train_sliding_temporal_embeddings.csv",
        "train_supervised_states.csv"
    )

    export_states(
        "validation_sliding_temporal_embeddings.csv",
        "validation_supervised_states.csv"
    )

    export_states(
        "test_sliding_temporal_embeddings.csv",
        "test_supervised_states.csv"
    )

    print("\nGenerated Files :")
    print("train_supervised_states.csv")
    print("validation_supervised_states.csv")
    print("test_supervised_states.csv")

    print("\nSaved Model :")
    print(os.path.join(MODEL_DIR, "state_learner_sliding.pt"))

    print("\n" + "=" * 60)
    print("STATE LEARNER COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ==========================================================

if __name__ == "__main__":
    main()