"""Experiment 7: how many labelled fish from the new farm are needed?

Source = SA (unique images, shared classes). Target = MatsyaDx (shared classes).
For k = 0,1,2,3,4 labelled target fish per class (all photos of those fish), train on
source + those photos, test on the remaining target fish. For conformal (90 %), a
second, disjoint set of k labelled fish per class is used only for calibration (never
for training), and those fish are also excluded from the test set.
20 random draws of fish per k. Frozen features, logistic regression.
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")
import os as _os
SAG = _os.environ.get("SA_GROUP", "group")  # "group" = pHash, "group_strict" = pHash + DINOv2 cos>=0.85
SFX = "" if SAG == "group" else "_strict"

R = _ROOT + "/results"
SA_NAMES = {0: "red", 1: "aeromoniasis", 2: "gill", 3: "saprolegniasis", 4: "healthy", 5: "parasitic", 6: "whitetail"}
MD_NAMES = {"Bacterial Gill Diseases": "gill", "Bacterial Red Diseases": "red",
            "Epizootic Ulcerative Syndrome (EUS)": "eus", "Healthy Fish": "healthy"}
SHARED = ["gill", "red", "healthy"]
sa = pd.read_csv(f"{R}/southasia_meta.csv"); sa["cls"] = sa.label.map(SA_NAMES)
md = pd.read_csv(f"{R}/matsyadx_meta.csv"); md["cls"] = md.health_condition.map(MD_NAMES)
sa_u = sa.groupby(SAG).head(1).index.values
s_idx = np.array([i for i in sa_u if sa.cls[i] in SHARED])
mdk = md[md.cls.isin(SHARED)]
fish = {c: mdk[mdk.cls == c].group.unique() for c in SHARED}


def conformal(clf, Xc, yc, Xt, yt, alpha=0.1):
    cls = list(clf.classes_)
    s = 1 - clf.predict_proba(Xc)[np.arange(len(yc)), [cls.index(c) for c in yc]]
    q = np.quantile(s, min(1.0, np.ceil((len(s) + 1) * (1 - alpha)) / len(s)), method="higher")
    sets = clf.predict_proba(Xt) >= 1 - q
    return sets[np.arange(len(yt)), [cls.index(c) for c in yt]].mean(), sets.sum(1).mean()


out = {}
for bb in ["dinov2s", "resnet50"]:
    XS = normalize(np.load(f"{R}/feat_southasia_{bb}.npy")); XM = normalize(np.load(f"{R}/feat_matsyadx_{bb}.npy"))
    ys, ym = sa.cls.values, md.cls.values
    res = {}
    for k in [0, 1, 2, 3, 4]:  # healthy has only 10 fish: 2k <= 8 keeps >= 2 for testing
        f1s, f1_tonly, covs, sizes = [], [], [], []
        for draw in range(20 if k else 1):
            rng = np.random.default_rng(draw)
            picks = {c: rng.permutation(fish[c]) for c in SHARED}
            lab_fish = set(sum([list(picks[c][:k]) for c in SHARED], []))
            cal_fish = set(sum([list(picks[c][k:2 * k]) for c in SHARED], []))
            lab = mdk.index[mdk.group.isin(lab_fish)].values
            cal = mdk.index[mdk.group.isin(cal_fish)].values
            test = mdk.index[~mdk.group.isin(lab_fish | cal_fish)].values
            Xtr = np.r_[XS[s_idx], XM[lab]]; ytr = np.r_[ys[s_idx], ym[lab]]
            # up-weight target photos so a few fish are not drowned by the source set
            wt = np.r_[np.ones(len(s_idx)), np.full(len(lab), len(s_idx) / max(len(lab), 1))]
            clf = LogisticRegression(max_iter=3000).fit(Xtr, ytr, sample_weight=wt)
            f1s.append(f1_score(ym[test], clf.predict(XM[test]), average="macro"))
            if k:
                c2 = LogisticRegression(max_iter=3000).fit(XM[lab], ym[lab])
                f1_tonly.append(f1_score(ym[test], c2.predict(XM[test]), average="macro"))
                # conformal calibrated on k *separate* labelled target fish (not used in training)
                cv, sz = conformal(clf, XM[cal], ym[cal], XM[test], ym[test])
                covs.append(cv); sizes.append(sz)
        r = {"macroF1_source+target": [round(np.mean(f1s), 4), round(np.std(f1s), 4)]}
        if k:
            r["macroF1_target_only"] = [round(np.mean(f1_tonly), 4), round(np.std(f1_tonly), 4)]
            r["conformal90_calib_on_k_other_fish"] = {"coverage": round(float(np.mean(covs)), 4),
                                                      "avg_set_size": round(float(np.mean(sizes)), 3)}
        res[f"k={k}_fish_per_class"] = r
    out[bb] = res
print(json.dumps(out, indent=1))
json.dump(out, open(f"{R}/exp7_few_target_fish{SFX}.json", "w"), indent=1)
