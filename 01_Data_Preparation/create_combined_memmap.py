import os
import numpy as np

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"
DATA = os.path.join(BASE, "data", "processed", "sliding_memmap")

for split in ["train_sliding", "validation_sliding", "test_sliding"]:

    print(f"\nProcessing {split}")

    X = np.load(os.path.join(DATA, f"{split}_X.npy"), mmap_mode="r")
    M = np.load(os.path.join(DATA, f"{split}_M.npy"), mmap_mode="r")

    N = len(X)

    out = np.lib.format.open_memmap(
        os.path.join(DATA, f"{split}_XM.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(N, 24, 80)
    )

    batch = 2000

    for i in range(0, N, batch):
        j = min(i + batch, N)
        out[i:j] = np.concatenate([X[i:j], M[i:j]], axis=2)
        print(f"{j}/{N}", end="\r")

    del out
    print("Completed")

print("\nAll Done")