import os
import numpy as np
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

SPLIT_DIR = os.path.join(BASE, "data", "processed", "splits")
OUT_DIR = os.path.join(BASE, "data", "processed", "sliding_lstm")

os.makedirs(OUT_DIR, exist_ok=True)

WINDOW = 24

# ==========================================================
# FUNCTION
# ==========================================================

def create_sliding_windows(csv_name, save_name):

    print(f"\nLoading {csv_name}")

    df = pd.read_csv(os.path.join(SPLIT_DIR, csv_name))

    # Training has SequenceID, Validation/Test have PatientID
    if "SequenceID" in df.columns:
        id_col = "SequenceID"
    else:
        id_col = "PatientID"

    exclude = ["PatientID", "SequenceID", "SepsisLabel"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X_list = []
    M_list = []
    y_list = []
    patient_ids = []
    window_ids = []

    for pid, patient in df.groupby(id_col, sort=False):

        patient = patient.reset_index(drop=True)

        values = patient[feature_cols].astype(np.float32).values
        mask = (~np.isnan(values)).astype(np.float32)
        values = np.nan_to_num(values, nan=0.0)

        n = len(values)

        # Skip patients with fewer than 24 hours
        if n < WINDOW:
            continue

        # Sliding windows
        for start in range(0, n - WINDOW + 1):

            end = start + WINDOW

            X = values[start:end]
            M = mask[start:end]

            # Label = whether sepsis occurs within this window
            label = int(patient.iloc[start:end]["SepsisLabel"].max())

            X_list.append(X)
            M_list.append(M)
            y_list.append(label)
            patient_ids.append(str(pid))
            window_ids.append(start)

    X = np.array(X_list, dtype=np.float32)
    M = np.array(M_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    patient_ids = np.array(patient_ids)
    window_ids = np.array(window_ids)

    np.savez_compressed(
        os.path.join(OUT_DIR, save_name),
        X=X,
        M=M,
        y=y,
        patient_ids=patient_ids,
        window_ids=window_ids
    )

    print("Windows :", len(X))
    print("Shape   :", X.shape)
    print("Saved   :", save_name)


# ==========================================================
# MAIN
# ==========================================================

print("=" * 60)
print("SLIDING WINDOW LSTM PREPARATION")
print("=" * 60)

create_sliding_windows("train_balanced.csv", "train_sliding.npz")
create_sliding_windows("validation.csv", "validation_sliding.npz")
create_sliding_windows("test.csv", "test_sliding.npz")

print("\nCompleted Successfully.")