"""Experiment 1: how much does duplicate leakage inflate accuracy on the South Asia set?

A  official split (HF/Kaggle train vs test, duplicates allowed across them)
B  official split, but test images that have a near-duplicate in train removed
C  5-fold StratifiedGroupKFold on near-duplicate groups (no group in both folds)
Classifier: L2-normalised frozen features -> logistic regression.
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import json, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")
import os as _os
SAG = _os.environ.get("SA_GROUP", "group")  # "group" = pHash, "group_strict" = pHash + DINOv2 cos>=0.85
SFX = "" if SAG == "group" else "_strict"

R = _ROOT + "/results"
meta = pd.read_csv(f"{R}/southasia_meta.csv")
y, g = meta.label.values, meta[SAG].values


def fit_eval(X, tr, te):
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr])
    p = clf.predict(X[te])
    return accuracy_score(y[te], p), f1_score(y[te], p, average="macro")


out = {}
for bb in ["dinov2s", "resnet50"]:
    X = normalize(np.load(f"{R}/feat_southasia_{bb}.npy"))
    tr = np.where(meta.split == "train")[0]
    te = np.where(meta.split == "test")[0]
    train_groups = set(g[tr])
    te_clean = np.array([i for i in te if g[i] not in train_groups])
    a = fit_eval(X, tr, te)
    b = fit_eval(X, tr, te_clean)
    accs, f1s = [], []
    for seed in range(5):
        for ftr, fte in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            # also drop exact/near duplicates inside the test fold (keep one per group)
            fte = pd.Series(fte).groupby(g[fte]).first().values
            acc, f1 = fit_eval(X, ftr, fte)
            accs.append(acc); f1s.append(f1)
    out[bb] = {
        "A_official_split": {"acc": a[0], "macroF1": a[1], "n_test": len(te)},
        "B_official_minus_leaky": {"acc": b[0], "macroF1": b[1], "n_test": len(te_clean)},
        "C_group_kfold_5x5": {"acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                              "macroF1": float(np.mean(f1s)), "macroF1_sd": float(np.std(f1s))},
    }
print(json.dumps(out, indent=2))
json.dump(out, open(f"{R}/exp1_within_southasia{SFX}.json", "w"), indent=2)
