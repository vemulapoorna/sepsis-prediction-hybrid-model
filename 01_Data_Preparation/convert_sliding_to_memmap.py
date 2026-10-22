import os
import numpy as np

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

SRC = os.path.join(BASE, "data", "processed", "sliding_lstm")
DST = os.path.join(BASE, "data", "processed", "sliding_memmap")

os.makedirs(DST, exist_ok=True)

# ==========================================================
# CONVERT FUNCTION
# ==========================================================

def convert(split_name):

    print(f"\nConverting {split_name}")

    data = np.load(
        os.path.join(SRC, f"{split_name}.npz"),
        allow_pickle=True
    )

    X = data["X"]
    M = data["M"]
    y = data["y"]
    patient = data["patient_ids"]
    window = data["window_ids"]

    print("Shape:", X.shape)

    np.save(os.path.join(DST, f"{split_name}_X.npy"), X)
    np.save(os.path.join(DST, f"{split_name}_M.npy"), M)
    np.save(os.path.join(DST, f"{split_name}_y.npy"), y)
    np.save(os.path.join(DST, f"{split_name}_patient.npy"), patient)
    np.save(os.path.join(DST, f"{split_name}_window.npy"), window)

    print("Saved.")


# ==========================================================
# MAIN
# ==========================================================

print("="*60)
print("CONVERT SLIDING DATA TO MEMORY MAP")
print("="*60)

convert("train_sliding")
convert("validation_sliding")
convert("test_sliding")

print("\nCompleted Successfully.")