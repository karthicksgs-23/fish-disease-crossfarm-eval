"""Experiment 6: can unlabelled target images repair cross-dataset failure? (frozen features)

Alignment (uses target images, never target labels):
  none      plain L2-normalised features
  center    subtract each domain's own mean, re-normalise
  coral     CORAL: whiten source covariance, re-colour with target covariance
Conformal under shift (target 90 %):
  standard  split conformal calibrated on source
  weighted  Tibshirani et al. (2019) covariate-shift weighted conformal; weights from a
            source-vs-target domain classifier fitted on half the unlabelled target,
            coverage measured on the other half
Unknown detection: rank-average of energy + kNN ("combined").
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.special import logsumexp, softmax
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
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
m_idx = np.where(md.cls.isin(SHARED))[0]
eus = np.where(md.cls == "eus")[0]


def coral(Xs, Xt, eps=1.0):
    cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xs.shape[1])
    A = np.real(np.linalg.inv(sqrtm(cs)) @ sqrtm(ct))
    return (Xs - Xs.mean(0)) @ A + Xt.mean(0)


def align(method, Xs, Xt):
    if method == "none":
        return Xs, Xt
    if method == "center":
        return normalize(Xs - Xs.mean(0)), normalize(Xt - Xt.mean(0))
    if method == "coral":
        return normalize(coral(Xs, Xt)), Xt
    raise ValueError


def wquantile(scores, weights, w_test, q):
    """weighted (1-alpha) quantile with the test point's own weight placed at +inf"""
    order = np.argsort(scores); s = scores[order]; w = weights[order]
    p = np.append(w, w_test) / (w.sum() + w_test)
    cum = np.cumsum(p[:-1])
    k = np.searchsorted(cum, q)
    return s[k] if k < len(s) else np.inf


def conformal_eval(clf, Xcal, ycal, Xte, yte, wcal=None, wte=None, alpha=0.10):
    cls = list(clf.classes_)
    s = 1 - clf.predict_proba(Xcal)[np.arange(len(ycal)), [cls.index(c) for c in ycal]]
    pt = clf.predict_proba(Xte)
    if wcal is None:
        qh = np.quantile(s, np.ceil((len(s) + 1) * (1 - alpha)) / len(s), method="higher")
        sets = pt >= 1 - qh
    else:
        sets = np.array([pt[i] >= 1 - wquantile(s, wcal, wte[i], 1 - alpha) for i in range(len(pt))])
    cov = sets[np.arange(len(yte)), [cls.index(c) for c in yte]].mean()
    return {"coverage": round(float(cov), 4), "avg_set_size": round(float(sets.sum(1).mean()), 3)}


out = {}
for bb in ["dinov2s", "resnet50"]:
    XS = normalize(np.load(f"{R}/feat_southasia_{bb}.npy")); XM = normalize(np.load(f"{R}/feat_matsyadx_{bb}.npy"))
    res = {}
    for direction, (Xsrc, ysrc, Xtgt, ytgt) in {
        "SA_to_MD": (XS[s_idx], sa.cls.values[s_idx], XM[m_idx], md.cls.values[m_idx]),
        "MD_to_SA": (XM[m_idx], md.cls.values[m_idx], XS[s_idx], sa.cls.values[s_idx]),
    }.items():
        res[direction] = {}
        for meth in ["none", "center", "coral"]:
            A, B = align(meth, Xsrc, Xtgt)
            p = LogisticRegression(max_iter=3000).fit(A, ysrc).predict(B)
            res[direction][meth] = {"macroF1": round(f1_score(ytgt, p, average="macro"), 4),
                                    "recall": {c: round(float((p[ytgt == c] == c).mean()), 3) for c in SHARED}}

    # ---- weighted conformal, SA -> MD (with centering, the best simple alignment)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(s_idx)); fit_i, cal_i = perm[: len(perm) // 2], perm[len(perm) // 2:]
    tperm = rng.permutation(len(m_idx)); t_dom, t_eval = tperm[: len(tperm) // 2], tperm[len(tperm) // 2:]
    conf = {}
    for meth in ["none", "center"]:
        Xs_a, Xt_a = align(meth, XS[s_idx], XM[m_idx])
        ys = sa.cls.values[s_idx]; yt = md.cls.values[m_idx]
        clf = LogisticRegression(max_iter=3000).fit(Xs_a[fit_i], ys[fit_i])
        conf[f"{meth}_standard"] = conformal_eval(clf, Xs_a[cal_i], ys[cal_i], Xt_a[t_eval], yt[t_eval])
        dom = LogisticRegression(max_iter=3000, C=0.1).fit(
            np.r_[Xs_a[cal_i], Xt_a[t_dom]], np.r_[np.zeros(len(cal_i)), np.ones(len(t_dom))])
        def w(X):
            pr = np.clip(dom.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3)
            return pr / (1 - pr)
        conf[f"{meth}_weighted"] = conformal_eval(clf, Xs_a[cal_i], ys[cal_i], Xt_a[t_eval], yt[t_eval],
                                                  wcal=w(Xs_a[cal_i]), wte=w(Xt_a[t_eval]))
    res["conformal_SA_to_MD_target90"] = conf

    # ---- combined unknown score: EUS vs known MD images, SA-trained (centered) model
    Xs_a, Xt_a = align("center", XS[s_idx], XM[np.r_[m_idx, eus]])
    clf = LogisticRegression(max_iter=3000).fit(Xs_a, sa.cls.values[s_idx])
    z = clf.decision_function(Xt_a)
    en = -logsumexp(z, 1); knn = 1 - np.sort(Xt_a @ Xs_a.T, 1)[:, -10:].mean(1)
    comb = (rankdata(en) + rankdata(knn)) / 2
    lab = np.r_[np.zeros(len(m_idx)), np.ones(len(eus))]
    res["EUS_unknown_AUROC_centered"] = {"energy": round(roc_auc_score(lab, en), 4),
                                         "kNN": round(roc_auc_score(lab, knn), 4),
                                         "combined": round(roc_auc_score(lab, comb), 4)}
    out[bb] = res
print(json.dumps(out, indent=1))
json.dump(out, open(f"{R}/exp6_shift_fixes{SFX}.json", "w"), indent=1)
