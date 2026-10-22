import os
import numpy as np
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

DBN_DIR = os.path.join(BASE, "data", "processed", "dbn")

train = pd.read_csv(
    os.path.join(DBN_DIR, "train_supervised_states.csv")
)

print("="*60)
print("SUPERVISED TEMPORAL DYNAMIC BAYESIAN NETWORK")
print("="*60)

# ==========================================================
# SORT TEMPORALLY
# ==========================================================

train = train.sort_values(
    ["PatientID","WindowID"]
).reset_index(drop=True)

# ==========================================================
# STATE DISTRIBUTION
# ==========================================================

state_summary = (
    train.groupby("State")
    .agg(
        Windows=("State","count"),
        SepsisCases=("SepsisLabel","sum"),
        SepsisRate=("SepsisLabel","mean")
    )
)

print("\nHidden State Summary")
print(state_summary)

# ==========================================================
# TRANSITION MATRIX
# ==========================================================

N = 4

transition = np.ones((N,N))

for patient_id, patient in train.groupby("PatientID"):

    states = patient["State"].values

    if len(states) < 2:
        continue

    for i in range(len(states)-1):

        s1 = int(states[i])
        s2 = int(states[i+1])

        transition[s1,s2] += 1

transition = transition / transition.sum(axis=1, keepdims=True)

transition_df = pd.DataFrame(
    transition,
    index=[f"State_{i}" for i in range(N)],
    columns=[f"Next_{i}" for i in range(N)]
)

print("\nTransition Matrix")
print(transition_df.round(4))

# ==========================================================
# EMISSION PROBABILITY
# ==========================================================

emission = (
    train.groupby("State")["SepsisLabel"]
    .mean()
    .reindex(range(N), fill_value=0)
)

emission_df = emission.to_frame("Probability")

print("\nEmission Probability")
print(emission_df.round(4))

# ==========================================================
# INITIAL PROBABILITY
# ==========================================================

first_states = (
    train.groupby("PatientID")
    .first()["State"]
)

pi = np.zeros(N)

for s in first_states:
    pi[int(s)] += 1

pi = pi / pi.sum()

pi_df = pd.DataFrame({
    "State":[0,1,2,3],
    "InitialProbability":pi
})

print("\nInitial State Distribution")
print(pi_df.round(4))

# ==========================================================
# SAVE
# ==========================================================

state_summary.to_csv(
    os.path.join(DBN_DIR,"state_summary_supervised.csv")
)

transition_df.to_csv(
    os.path.join(DBN_DIR,"transition_matrix_supervised.csv")
)

emission_df.to_csv(
    os.path.join(DBN_DIR,"emission_probability_supervised.csv")
)

pi_df.to_csv(
    os.path.join(DBN_DIR,"initial_probability.csv"),
    index=False
)

print("\nSaved Files")
print("state_summary_supervised.csv")
print("transition_matrix_supervised.csv")
print("emission_probability_supervised.csv")
print("initial_probability.csv")

print("\nDBN CREATED SUCCESSFULLY")