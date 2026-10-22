import os
import numpy as np
import pandas as pd
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
OUT_DIR = os.path.join(BASE, "data", "processed", "dbn")

os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "sliding_lstm_encoder.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("EXTRACTING TEMPORAL EMBEDDINGS")
print("="*60)

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

        self.patient = np.load(
            os.path.join(DATA_DIR, f"{prefix}_patient.npy"),
            mmap_mode="r"
        )

        self.window = np.load(
            os.path.join(DATA_DIR, f"{prefix}_window.npy"),
            mmap_mode="r"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return (
            torch.from_numpy(self.X[idx].copy()).float(),
            self.y[idx],
            self.patient[idx],
            self.window[idx]
        )

# ==========================================================
# MODEL
# ==========================================================

class TemporalAttention(nn.Module):

    def __init__(self, hidden):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(hidden*2,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self,x):

        w=torch.softmax(self.score(x),dim=1)
        context=torch.sum(w*x,dim=1)

        return context


class SlidingEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm=nn.LayerNorm(80)

        self.lstm=nn.LSTM(
            80,64,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )

        self.attn=TemporalAttention(64)

        self.embed=nn.Linear(128,64)

        self.drop=nn.Dropout(0.3)

        self.cls=nn.Linear(64,1)

    def forward(self,x):

        x=self.norm(x)

        out,_=self.lstm(x)

        context=self.attn(out)

        z=torch.relu(self.embed(context))

        z=self.drop(z)

        logit=self.cls(z)

        return logit.squeeze(),z


model=SlidingEncoder().to(device)

model.load_state_dict(torch.load(MODEL_PATH,map_location=device))

model.eval()

print("Encoder Loaded Successfully")

# ==========================================================
# EXTRACTION FUNCTION
# ==========================================================

def extract(prefix):

    ds=SlidingDataset(prefix)

    loader=DataLoader(
        ds,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    embeddings=[]
    labels=[]
    patients=[]
    windows=[]

    with torch.no_grad():

        for X,y,p,w in tqdm(loader,ncols=100,desc=prefix):

            X=X.to(device)

            _,z=model(X)

            embeddings.append(z.cpu().numpy())

            labels.extend(y.numpy())

            patients.extend(p)

            windows.extend(w.numpy())

    embeddings=np.vstack(embeddings)

    cols=[f"z{i+1}" for i in range(64)]

    df=pd.DataFrame(embeddings,columns=cols)

    df.insert(0,"WindowID",windows)
    df.insert(0,"PatientID",patients)

    df["SepsisLabel"]=labels

    save_path=os.path.join(
        OUT_DIR,
        f"{prefix}_temporal_embeddings.csv"
    )

    df.to_csv(save_path,index=False)

    print("Saved:",save_path)
    print("Shape:",df.shape)


# ==========================================================
# RUN
# ==========================================================

extract("train_sliding")
extract("validation_sliding")
extract("test_sliding")

print("\nCompleted Successfully.")