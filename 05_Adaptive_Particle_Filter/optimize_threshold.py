import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

DBN_DIR = os.path.join(BASE, "data", "processed", "dbn")
RESULT_DIR = os.path.join(BASE, "data", "processed", "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================================
# LOAD DBN PARAMETERS
# ==========================================================

transition = pd.read_csv(
    os.path.join(DBN_DIR, "transition_matrix_supervised.csv"),
    index_col=0
).values

emission = pd.read_csv(
    os.path.join(DBN_DIR, "emission_probability_supervised.csv"),
    index_col=0
)["Probability"].values

initial = pd.read_csv(
    os.path.join(DBN_DIR, "initial_probability.csv")
)["InitialProbability"].values

NUM_PARTICLES = 200
NUM_STATES = 4

# ==========================================================
# PARTICLE FILTER FUNCTION
# ==========================================================

def run_particle_filter(csv_file):

    df = pd.read_csv(csv_file)

    df = df.sort_values(["PatientID","WindowID"])

    patient_results = []

    groups = list(df.groupby("PatientID"))

    for pid, patient in tqdm(groups, total=len(groups), desc=os.path.basename(csv_file)):

        particles = np.random.choice(
            NUM_STATES,
            size=NUM_PARTICLES,
            p=initial
        )

        weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

        patient = patient.sort_values("WindowID")

        for _, row in patient.iterrows():

            # ---------------- Transition ----------------

            probs = transition[particles]

            rnd = np.random.rand(NUM_PARTICLES)

            cdf = np.cumsum(probs, axis=1)

            particles = (rnd[:, None] > cdf).sum(axis=1)

            # ---------------- Observation ----------------

            obs = int(row["State"])

            weights *= transition[particles, obs]

            if row["SepsisLabel"] == 1:
                weights *= emission[particles]
            else:
                weights *= (1 - emission[particles])

            weights += 1e-12
            weights /= weights.sum()

            # ---------------- Adaptive Resampling ----------------

            ess = 1 / np.sum(weights ** 2)

            if ess < NUM_PARTICLES * 0.5:

                idx = np.random.choice(
                    NUM_PARTICLES,
                    NUM_PARTICLES,
                    p=weights
                )

                particles = particles[idx]

                weights[:] = 1 / NUM_PARTICLES

        prob = np.sum(emission[particles] * weights)

        patient_results.append([
            pid,
            int(patient["SepsisLabel"].max()),
            prob
        ])

    return pd.DataFrame(
        patient_results,
        columns=["PatientID","TrueLabel","Probability"]
    )

# ==========================================================
# FIND BEST THRESHOLD
# ==========================================================

def find_best_threshold(df):

    y_true = df["TrueLabel"].values
    y_prob = df["Probability"].values

    best_t = 0.50
    best_f1 = -1

    for t in np.arange(0.05, 0.96, 0.01):

        pred = (y_prob >= t).astype(int)

        score = f1_score(y_true, pred)

        if score > best_f1:

            best_f1 = score
            best_t = t

    return round(best_t,2), best_f1

# ==========================================================
# EVALUATION
# ==========================================================

def evaluate(df, threshold):

    y_true = df["TrueLabel"].values
    y_prob = df["Probability"].values

    y_pred = (y_prob >= threshold).astype(int)

    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    spec = tn / (tn + fp)

    metrics = {
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "F1": f1,
        "Accuracy": acc,
        "Sensitivity": rec,
        "Specificity": spec,
        "Precision": pre
    }

    return metrics

# ==========================================================
# MAIN
# ==========================================================

print("="*60)
print("THRESHOLD OPTIMIZATION")
print("="*60)

print("\nRunning Validation Particle Filter...\n")

val_pred = run_particle_filter(
    os.path.join(DBN_DIR, "validation_supervised_states.csv")
)

best_threshold, best_f1 = find_best_threshold(val_pred)

print("\nBest Threshold Found :", best_threshold)
print("Validation F1        :", round(best_f1,4))

print("\nRunning Test Particle Filter...\n")

test_pred = run_particle_filter(
    os.path.join(DBN_DIR, "test_supervised_states.csv")
)

metrics = evaluate(test_pred, best_threshold)

print("\n" + "="*60)
print("FINAL OPTIMIZED RESULTS")
print("="*60)

for k, v in metrics.items():
    print(f"{k:15}: {v:.4f}")

print(f"{'Threshold':15}: {best_threshold:.2f}")

# ==========================================================
# SAVE
# ==========================================================

test_pred["PredictedLabel"] = (
    test_pred["Probability"] >= best_threshold
).astype(int)

test_pred.to_csv(
    os.path.join(RESULT_DIR, "optimized_patient_predictions.csv"),
    index=False
)

pd.DataFrame([metrics]).assign(
    Threshold=best_threshold
).to_csv(
    os.path.join(RESULT_DIR, "optimized_metrics.csv"),
    index=False
)

print("\nSaved Files")
print("optimized_metrics.csv")
print("optimized_patient_predictions.csv")