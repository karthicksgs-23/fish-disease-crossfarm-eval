"""Frozen DINOv2-S + logistic regression on the same five sa_openset folds as train_lora.py
(strict groups, Parasitic held out of training), for a paired frozen-vs-LoRA comparison."""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, pandas as pd
from scipy.special import logsumexp, softmax
from scipy.stats import wilcoxon
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
sys.stdout.reconfigure(encoding="utf-8")
R = _ROOT + "/results"
SA_NAMES = {0: "red", 1: "aeromoniasis", 2: "gill", 3: "saprolegniasis", 4: "healthy", 5: "parasitic", 6: "whitetail"}
sa = pd.read_csv(f"{R}/southasia_meta.csv"); y = sa.label.map(SA_NAMES).values; g = sa.group_strict.values
sa_u = sa.groupby("group_strict").head(1).index.values
def fpr95(sk, su):
    return float((sk >= np.percentile(su, 5)).mean())
allres = {}
for bb in ["dinov2s", "resnet50"]:
    X = normalize(np.load(f"{R}/feat_southasia_{bb}.npy"))
    res = {k: [] for k in ["MSP", "energy", "kNN", "Mahalanobis", "kNN_FPR95", "f1", "acc", "n_test", "n_unknown"]}
    for trg, teg in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(sa_u, y[sa_u], g[sa_u]):
        trs = set(g[sa_u[trg]])
        tr = np.array([i for i in range(len(sa)) if g[i] in trs and y[i] != "parasitic"]); te = sa_u[teg]
        clf = LogisticRegression(max_iter=3000).fit(X[tr], y[tr]); z = clf.decision_function(X[te])
        unk = (y[te] == "parasitic").astype(int); kn = unk == 0; p = clf.predict(X[te])
        res["f1"].append(f1_score(y[te][kn], p[kn], average="macro")); res["acc"].append(accuracy_score(y[te][kn], p[kn]))
        res["MSP"].append(roc_auc_score(unk, 1 - softmax(z, 1).max(1)))
        res["energy"].append(roc_auc_score(unk, -logsumexp(z, 1)))
        knn = 1 - np.sort(X[te] @ X[tr].T, 1)[:, -10:].mean(1)
        res["kNN"].append(roc_auc_score(unk, knn)); res["kNN_FPR95"].append(fpr95(knn[kn], knn[~kn]))
        cl = np.unique(y[tr]); mu = {c: X[tr][y[tr] == c].mean(0) for c in cl}
        P = LedoitWolf().fit(np.vstack([X[tr][y[tr] == c] - mu[c] for c in cl])).precision_
        res["Mahalanobis"].append(roc_auc_score(unk, np.min([np.einsum("ij,jk,ik->i", X[te] - mu[c], P, X[te] - mu[c]) for c in cl], 0)))
        res["n_test"].append(len(te)); res["n_unknown"].append(int(unk.sum()))
    allres[bb] = res
res = allres["dinov2s"]
out = {"resnet50": {k: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4)} for k, v in allres["resnet50"].items()}}
out.update({k: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4), "values": [round(float(x), 4) for x in v]} for k, v in res.items()})
L = [json.load(open(f"{R}/lora_sa_openset_strict_f{f}_s0.json"))["SA_open_test"] for f in range(5)]
lora = {"f1": [d["macroF1_known"] for d in L], "MSP": [d["unknown_AUROC"]["MSP"] for d in L], "kNN": [d["unknown_AUROC"]["kNN"] for d in L]}
out["paired_lora_minus_frozen"] = {}
for k in lora:
    r = wilcoxon(lora[k], res[k], method="exact")
    out["paired_lora_minus_frozen"][k] = {"median_diff": round(float(np.median(np.array(lora[k]) - np.array(res[k]))), 4),
                                         "lora_wins": int(sum(a > b for a, b in zip(lora[k], res[k]))), "p": float(f"{r.pvalue:.3g}")}
print(json.dumps(out, indent=1)); json.dump(out, open(f"{R}/exp9_openset_frozen_5fold.json", "w"), indent=1)
