"""Experiment 8: fish-level (cluster) conformal prediction on MatsyaDx-BD (frozen DINOv2-S).

Repeated 100 times: within each class, fish are split 50 % train / 25 % calibration /
25 % test (at least one fish per part). A logistic regression is trained on the train
photos. Three ways to calibrate a 90 % prediction set (alpha = 0.10):
  photo     every calibration photo is one score (assumes photos are exchangeable)
  fish_max  one score per calibration fish = its worst (largest) photo score;
            threshold applied to every photo of a test fish
  fish_avg  probabilities averaged over a fish's photos; one score and one set per fish
Reported: photo-level coverage, fish-level coverage (all photos of the fish covered,
or the fish-level set for fish_avg), mean set size, and the share of repetitions
whose coverage fell below 90 %.
Part B repeats the cross-farm setting of Experiment 7 (SA-trained model, k labelled
MatsyaDx-BD calibration fish per shared class) with photo vs fish_max calibration.
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")
R = _ROOT + "/results"
ALPHA = 0.10
MD_NAMES = {"Bacterial Gill Diseases": "gill", "Bacterial Red Diseases": "red",
            "Epizootic Ulcerative Syndrome (EUS)": "eus", "Healthy Fish": "healthy"}
SA_NAMES = {0: "red", 1: "aeromoniasis", 2: "gill", 3: "saprolegniasis", 4: "healthy", 5: "parasitic", 6: "whitetail"}
SHARED = ["gill", "red", "healthy"]
md = pd.read_csv(f"{R}/matsyadx_meta.csv"); md["cls"] = md.health_condition.map(MD_NAMES)
XM = normalize(np.load(f"{R}/feat_matsyadx_dinov2s.npy"))


def qhat(scores):
    n = len(scores)
    lev = np.ceil((n + 1) * (1 - ALPHA)) / n
    return np.inf if lev > 1 else np.quantile(scores, lev, method="higher")


def evaluate(P, y, fish, classes, method, cal):
    """cal = (P_cal, y_cal, fish_cal). Returns photo coverage, fish coverage, mean set size."""
    Pc, yc, fc = cal
    ci = {c: i for i, c in enumerate(classes)}
    sc = 1 - Pc[np.arange(len(yc)), [ci[c] for c in yc]]
    if method == "photo":
        q = qhat(sc)
    elif method == "fish_max":
        q = qhat(pd.Series(sc).groupby(fc).max().values)
    elif method == "fish_avg":
        dfc = pd.DataFrame(Pc).groupby(fc).mean()
        lab = pd.Series(yc).groupby(fc).first().loc[dfc.index]
        q = qhat(1 - dfc.values[np.arange(len(lab)), [ci[c] for c in lab]])
        dft = pd.DataFrame(P).groupby(fish).mean()
        labt = pd.Series(y).groupby(fish).first().loc[dft.index]
        sets = dft.values >= 1 - q
        cov = sets[np.arange(len(labt)), [ci[c] for c in labt]]
        return float(cov.mean()), float(cov.mean()), float(sets.sum(1).mean())
    sets = P >= 1 - q
    hit = sets[np.arange(len(y)), [ci[c] for c in y]]
    fish_cov = pd.Series(hit).groupby(fish).all().mean()
    return float(hit.mean()), float(fish_cov), float(sets.sum(1).mean())


# ---------------- Part A: in-domain MatsyaDx-BD, 4 classes
classes = sorted(md.cls.unique())
fish_by_cls = {c: md[md.cls == c].group.unique() for c in classes}
res = {m: [] for m in ["photo", "fish_max", "fish_avg"]}
for rep in range(100):
    rng = np.random.default_rng(rep)
    tr_f, ca_f, te_f = [], [], []
    for c in classes:
        f = rng.permutation(fish_by_cls[c]); n = len(f)
        n_ca = max(1, round(0.25 * n)); n_te = max(1, round(0.25 * n))
        te_f += list(f[:n_te]); ca_f += list(f[n_te:n_te + n_ca]); tr_f += list(f[n_te + n_ca:])
    tr = md.group.isin(tr_f).values; ca = md.group.isin(ca_f).values; te = md.group.isin(te_f).values
    clf = LogisticRegression(max_iter=3000).fit(XM[tr], md.cls.values[tr])
    assert list(clf.classes_) == classes
    cal = (clf.predict_proba(XM[ca]), md.cls.values[ca], md.group.values[ca])
    Pt = clf.predict_proba(XM[te])
    for m in res:
        res[m].append(evaluate(Pt, md.cls.values[te], md.group.values[te], classes, m, cal))
partA = {}
for m, v in res.items():
    v = np.array(v)
    partA[m] = {"photo_coverage": [round(v[:, 0].mean(), 4), round(v[:, 0].std(), 4)],
                "fish_coverage": [round(v[:, 1].mean(), 4), round(v[:, 1].std(), 4)],
                "mean_set_size": round(v[:, 2].mean(), 3),
                "share_reps_photo_cov_below_90": round(float((v[:, 0] < 0.9).mean()), 3),
                "share_reps_fish_cov_below_90": round(float((v[:, 1] < 0.9).mean()), 3)}
partA["n_calibration_fish_per_rep"] = len(ca_f)
partA["n_test_fish_per_rep"] = len(te_f)

# ---------------- Part B: cross-farm, SA-trained model, k labelled MD calibration fish per class
sa = pd.read_csv(f"{R}/southasia_meta.csv"); sa["cls"] = sa.label.map(SA_NAMES)
SAG = os.environ.get("SA_GROUP", "group_strict")
sa_u = sa.groupby(SAG).head(1).index.values
s_idx = np.array([i for i in sa_u if sa.cls[i] in SHARED])
XS = normalize(np.load(f"{R}/feat_southasia_dinov2s.npy"))
clf = LogisticRegression(max_iter=3000).fit(XS[s_idx], sa.cls.values[s_idx])
mdk = md[md.cls.isin(SHARED)]
fishk = {c: mdk[mdk.cls == c].group.unique() for c in SHARED}
cls3 = list(clf.classes_)
partB = {}
for k in [1, 2, 3, 4]:
    out = {m: [] for m in ["photo", "fish_max"]}
    for draw in range(50):
        rng = np.random.default_rng(draw)
        cal_f = sum([list(rng.choice(fishk[c], k, replace=False)) for c in SHARED], [])
        ca = mdk.group.isin(cal_f).values; te = ~ca
        cal = (clf.predict_proba(XM[mdk.index[ca]]), mdk.cls.values[ca], mdk.group.values[ca])
        Pt = clf.predict_proba(XM[mdk.index[te]])
        for m in out:
            out[m].append(evaluate(Pt, mdk.cls.values[te], mdk.group.values[te], cls3, m, cal))
    partB[f"k={k}"] = {m: {"photo_coverage": round(float(np.mean([x[0] for x in v])), 4),
                           "fish_coverage": round(float(np.mean([x[1] for x in v])), 4),
                           "mean_set_size": round(float(np.mean([x[2] for x in v])), 3)} for m, v in out.items()}
    partB[f"k={k}"]["n_cal_scores_fish_level"] = 3 * k

result = {"partA_in_domain_MD_4class_100reps": partA, "partB_cross_farm_SA_to_MD": partB}
print(json.dumps(result, indent=1))
json.dump(result, open(f"{R}/exp8_fish_conformal.json", "w"), indent=1)
