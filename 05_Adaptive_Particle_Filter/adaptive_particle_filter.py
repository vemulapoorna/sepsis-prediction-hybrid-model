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

BASE = r"C:\Users\Poorna\OneDrive\Documents\pr2\sepsis_probability_project"

DBN = os.path.join(BASE,"data","processed","dbn")
RESULTS = os.path.join(BASE,"data","processed","results")
os.makedirs(RESULTS,exist_ok=True)

transition = pd.read_csv(
    os.path.join(DBN,"transition_matrix_supervised.csv"),
    index_col=0
).values

emission = pd.read_csv(
    os.path.join(DBN,"emission_probability_supervised.csv"),
    index_col=0
)["Probability"].values

initial = pd.read_csv(
    os.path.join(DBN,"initial_probability.csv")
)["InitialProbability"].values

test = pd.read_csv(
    os.path.join(DBN,"test_supervised_states.csv")
)

test = test.sort_values(["PatientID","WindowID"])

print("="*60)
print("ADAPTIVE BAYESIAN PARTICLE FILTER")
print("="*60)
print("Patients :",test.PatientID.nunique())
print("Windows  :",len(test))

NUM_PARTICLES = 200
NUM_STATES = 4

results = []

groups = list(test.groupby("PatientID"))

for pid, patient in tqdm(groups, total=len(groups), desc="Filtering"):

    particles = np.random.choice(
        NUM_STATES,
        size=NUM_PARTICLES,
        p=initial
    )

    weights = np.ones(NUM_PARTICLES)/NUM_PARTICLES

    patient = patient.sort_values("WindowID")

    for _, row in patient.iterrows():

        # transition
        probs = transition[particles]
        rnd = np.random.rand(NUM_PARTICLES)
        cdf = np.cumsum(probs,axis=1)
        particles = (rnd[:,None] > cdf).sum(axis=1)

        # observation update
        obs = int(row.State)
        weights *= transition[particles,obs]

        if row.SepsisLabel==1:
            weights *= emission[particles]
        else:
            weights *= (1-emission[particles])

        weights += 1e-12
        weights /= weights.sum()

        # adaptive resampling
        ess = 1/np.sum(weights**2)

        if ess < NUM_PARTICLES*0.5:
            idx = np.random.choice(
                NUM_PARTICLES,
                NUM_PARTICLES,
                p=weights
            )
            particles = particles[idx]
            weights[:] = 1/NUM_PARTICLES

    prob = np.sum(emission[particles]*weights)

    results.append([pid,
                    int(patient.SepsisLabel.max()),
                    prob])

pred = pd.DataFrame(
    results,
    columns=["PatientID","TrueLabel","Probability"]
)

y_true = pred.TrueLabel.values
y_prob = pred.Probability.values
y_pred = (y_prob>=0.5).astype(int)

roc = roc_auc_score(y_true,y_prob)
pr = average_precision_score(y_true,y_prob)
acc = accuracy_score(y_true,y_pred)
pre = precision_score(y_true,y_pred)
rec = recall_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)

tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()
spec = tn/(tn+fp)

print("\nRESULTS")
print("-"*40)
print("ROC_AUC      :",round(roc,4))
print("PR_AUC       :",round(pr,4))
print("F1           :",round(f1,4))
print("Accuracy     :",round(acc,4))
print("Sensitivity  :",round(rec,4))
print("Specificity  :",round(spec,4))
print("Precision    :",round(pre,4))

pred.to_csv(
    os.path.join(RESULTS,"adaptive_particle_predictions.csv"),
    index=False
)

pd.DataFrame({
    "ROC_AUC":[roc],
    "PR_AUC":[pr],
    "F1":[f1],
    "Accuracy":[acc],
    "Sensitivity":[rec],
    "Specificity":[spec],
    "Precision":[pre]
}).to_csv(
    os.path.join(RESULTS,"adaptive_particle_metrics.csv"),
    index=False
)

print("\nSaved Successfully")