"""Experiments 2-5 (pilot), frozen features + logistic regression.

2  MatsyaDx-BD within-dataset: random image split vs specimen-grouped split
3  Cross-dataset on the 3 shared classes (gill, red, healthy): SA -> MD and MD -> SA
4  Open-set: is an unseen disease flagged as "unknown"?
     4a SA, Parasitic held out (group-aware)      4b SA-trained model, MD EUS images
   scores: max softmax prob (MSP), energy, kNN cosine distance, Mahalanobis
5  Split-conformal prediction (LAC, target 90 %) calibrated on SA, tested on SA and MD
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import json, sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupShuffleSplit
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")
import os as _os
SAG = _os.environ.get("SA_GROUP", "group")  # "group" = pHash, "group_strict" = pHash + DINOv2 cos>=0.85
SFX = "" if SAG == "group" else "_strict"

R = _ROOT + "/results"
SA_NAMES = {0: "red", 1: "aeromoniasis", 2: "gill", 3: "saprolegniasis", 4: "healthy",
            5: "parasitic", 6: "whitetail"}
MD_NAMES = {"Bacterial Gill Diseases": "gill", "Bacterial Red Diseases": "red",
            "Epizootic Ulcerative Syndrome (EUS)": "eus", "Healthy Fish": "healthy"}
SHARED = ["gill", "red", "healthy"]

sa = pd.read_csv(f"{R}/southasia_meta.csv")
sa["cls"] = sa.label.map(SA_NAMES)
md = pd.read_csv(f"{R}/matsyadx_meta.csv")
md["cls"] = md.health_condition.map(MD_NAMES)
assert md.cls.notna().all(), md.health_condition.unique()

# one image per near-duplicate group in SA so duplicates never inflate any score
sa_u = sa.groupby(SAG).head(1).index.values


def LR(X, y):
    return LogisticRegression(max_iter=3000, C=1.0).fit(X, y)


def scores_unknown(clf, Xtr, ytr, Xte):
    """higher = more likely unknown"""
    z = clf.decision_function(Xte)
    if z.ndim == 1:
        z = np.c_[-z, z]
    msp = 1 - softmax(z, 1).max(1)
    energy = -logsumexp(z, 1)
    sims = Xte @ Xtr.T
    knn = 1 - np.sort(sims, 1)[:, -10:].mean(1)
    lw = LedoitWolf().fit(np.vstack([Xtr[ytr == c] - Xtr[ytr == c].mean(0) for c in np.unique(ytr)]))
    P = lw.precision_
    maha = np.min([np.einsum("ij,jk,ik->i", Xte - Xtr[ytr == c].mean(0), P, Xte - Xtr[ytr == c].mean(0))
                   for c in np.unique(ytr)], 0)
    return {"MSP": msp, "energy": energy, "kNN": knn, "Mahalanobis": maha}


def fpr95(s_known, s_unk):
    thr = np.percentile(s_unk, 5)  # 95 % of unknowns flagged
    return float((s_known >= thr).mean())


def conformal(clf, Xcal, ycal, Xte, yte, alpha=0.10):
    cls = list(clf.classes_)
    pc = clf.predict_proba(Xcal)
    s = 1 - pc[np.arange(len(ycal)), [cls.index(c) for c in ycal]]
    q = np.quantile(s, np.ceil((len(s) + 1) * (1 - alpha)) / len(s), method="higher")
    pt = clf.predict_proba(Xte)
    sets = pt >= 1 - q
    cov = sets[np.arange(len(yte)), [cls.index(c) for c in yte]].mean()
    return {"coverage": float(cov), "avg_set_size": float(sets.sum(1).mean()),
            "empty_set_rate": float((sets.sum(1) == 0).mean())}


out = {}
for bb in ["dinov2s", "resnet50"]:
    Xs = normalize(np.load(f"{R}/feat_southasia_{bb}.npy"))
    Xm = normalize(np.load(f"{R}/feat_matsyadx_{bb}.npy"))
    res = {}

    # ---- 2: MatsyaDx random vs specimen-grouped split
    ym, gm = md.cls.values, md.group.values
    rnd, grp = [], []
    for seed in range(5):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xm, ym):
            rnd.append(f1_score(ym[te], LR(Xm[tr], ym[tr]).predict(Xm[te]), average="macro"))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xm, ym, gm):
            grp.append(f1_score(ym[te], LR(Xm[tr], ym[tr]).predict(Xm[te]), average="macro"))
    res["exp2_matsyadx_within"] = {"random_split_macroF1": [np.mean(rnd), np.std(rnd)],
                                   "specimen_split_macroF1": [np.mean(grp), np.std(grp)],
                                   "n_specimens": int(md.group.nunique())}

    # ---- 3: cross-dataset on shared classes
    s_idx = np.array([i for i in sa_u if sa.cls[i] in SHARED])
    m_idx = np.where(md.cls.isin(SHARED))[0]
    ys, ymm = sa.cls.values, md.cls.values
    cross = {}
    # in-domain reference (group-aware CV) for each side
    ref_s = [f1_score(ys[s_idx][te], LR(Xs[s_idx][tr], ys[s_idx][tr]).predict(Xs[s_idx][te]), average="macro")
             for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xs[s_idx], ys[s_idx], sa[SAG].values[s_idx])]
    ref_m = [f1_score(ymm[m_idx][te], LR(Xm[m_idx][tr], ymm[m_idx][tr]).predict(Xm[m_idx][te]), average="macro")
             for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xm[m_idx], ymm[m_idx], md.group.values[m_idx])]
    clf_s = LR(Xs[s_idx], ys[s_idx])
    clf_m = LR(Xm[m_idx], ymm[m_idx])
    p_sm = clf_s.predict(Xm[m_idx]); p_ms = clf_m.predict(Xs[s_idx])
    cross["SA_in_domain_macroF1"] = float(np.mean(ref_s))
    cross["MD_in_domain_macroF1"] = float(np.mean(ref_m))
    cross["SA_to_MD"] = {"acc": accuracy_score(ymm[m_idx], p_sm), "macroF1": f1_score(ymm[m_idx], p_sm, average="macro"),
                         "per_class_recall": {c: float((p_sm[ymm[m_idx] == c] == c).mean()) for c in SHARED},
                         "pred_distribution": pd.Series(p_sm).value_counts(normalize=True).round(3).to_dict()}
    cross["MD_to_SA"] = {"acc": accuracy_score(ys[s_idx], p_ms), "macroF1": f1_score(ys[s_idx], p_ms, average="macro"),
                         "per_class_recall": {c: float((p_ms[ys[s_idx] == c] == c).mean()) for c in SHARED}}
    res["exp3_cross_dataset"] = cross

    # ---- 4a: SA open-set, Parasitic held out
    gs = sa[SAG].values
    tr_g, te_g = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(sa_u, groups=gs[sa_u]))
    tr = np.array([i for i in sa_u[tr_g] if sa.cls[i] != "parasitic"])
    te = sa_u[te_g]
    clf = LR(Xs[tr], ys[tr])
    sc = scores_unknown(clf, Xs[tr], ys[tr], Xs[te])
    unk = (ys[te] == "parasitic").astype(int)
    res["exp4a_SA_parasitic_unknown"] = {k: {"AUROC": roc_auc_score(unk, v), "FPR95": fpr95(v[unk == 0], v[unk == 1])}
                                         for k, v in sc.items()}
    res["exp4a_SA_parasitic_unknown"]["parasitic_predicted_as"] = (
        pd.Series(clf.predict(Xs[te][unk == 1])).value_counts(normalize=True).round(3).to_dict())

    # ---- 4b: SA-trained (shared classes) model meets MatsyaDx EUS
    eus = np.where(md.cls == "eus")[0]
    known = m_idx
    sc = scores_unknown(clf_s, Xs[s_idx], ys[s_idx], Xm[np.r_[known, eus]])
    unk = np.r_[np.zeros(len(known)), np.ones(len(eus))]
    res["exp4b_MD_EUS_unknown_under_shift"] = {k: {"AUROC": roc_auc_score(unk, v)} for k, v in sc.items()}
    # and: does a shifted *known* class look unknown? SA test (known, in-domain) vs MD known
    te_in = sa_u[te_g][np.isin(ys[sa_u[te_g]], SHARED)]
    clf_tr = LR(Xs[np.array([i for i in sa_u[tr_g] if ys[i] in SHARED])], ys[np.array([i for i in sa_u[tr_g] if ys[i] in SHARED])])
    Xtr_k = Xs[np.array([i for i in sa_u[tr_g] if ys[i] in SHARED])]
    ytr_k = ys[np.array([i for i in sa_u[tr_g] if ys[i] in SHARED])]
    sc = scores_unknown(clf_tr, Xtr_k, ytr_k, np.r_[Xs[te_in], Xm[known]])
    lab = np.r_[np.zeros(len(te_in)), np.ones(len(known))]
    res["exp4c_dataset_shift_detectable"] = {k: {"AUROC_SAtest_vs_MD": roc_auc_score(lab, v)} for k, v in sc.items()}

    # ---- 5: conformal, calibrated on SA (shared classes), tested in-domain and on MD
    tr_k = np.array([i for i in sa_u[tr_g] if ys[i] in SHARED])
    rng = np.random.default_rng(0)
    perm = rng.permutation(tr_k)
    fit_i, cal_i = perm[: len(perm) // 2], perm[len(perm) // 2:]
    clf_c = LR(Xs[fit_i], ys[fit_i])
    res["exp5_conformal_90"] = {
        "SA_test_in_domain": conformal(clf_c, Xs[cal_i], ys[cal_i], Xs[te_in], ys[te_in]),
        "MD_shifted": conformal(clf_c, Xs[cal_i], ys[cal_i], Xm[known], ymm[known]),
    }
    out[bb] = res


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


out = clean(out)
print(json.dumps(out, indent=1))
json.dump(out, open(f"{R}/exp2to5_results{SFX}.json", "w"), indent=1)
