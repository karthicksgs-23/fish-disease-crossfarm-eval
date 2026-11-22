"""Collect multi-fold / multi-seed LoRA results and paired tests.

Paired comparisons:
  T1  MatsyaDx-BD frozen DINOv2: random image split vs split by fish (25 fold-seed pairs)
  T2  frozen DINOv2 vs LoRA, same folds (South Asia strict group split; MatsyaDx-BD by fish)
Wilcoxon signed-rank (two-sided, exact). With 5 pairs the smallest attainable p is 0.0625,
so these are reported as effect sizes with an exact p, not as significance claims.
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import glob, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")
R = _ROOT + "/results"
SA_NAMES = {0: "red", 1: "aeromoniasis", 2: "gill", 3: "saprolegniasis", 4: "healthy", 5: "parasitic", 6: "whitetail"}
MD_NAMES = {"Bacterial Gill Diseases": "gill", "Bacterial Red Diseases": "red",
            "Epizootic Ulcerative Syndrome (EUS)": "eus", "Healthy Fish": "healthy"}
sa = pd.read_csv(f"{R}/southasia_meta.csv"); sa["cls"] = sa.label.map(SA_NAMES)
md = pd.read_csv(f"{R}/matsyadx_meta.csv"); md["cls"] = md.health_condition.map(MD_NAMES)
XS = normalize(np.load(f"{R}/feat_southasia_dinov2s.npy")); XM = normalize(np.load(f"{R}/feat_matsyadx_dinov2s.npy"))


def ms(v):
    v = np.asarray(v, float)
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
            "n": int(len(v)), "values": [round(float(x), 4) for x in v]}


def wx(a, b):
    a, b = np.asarray(a), np.asarray(b)
    r = wilcoxon(a, b, alternative="two-sided", method="exact")
    return {"median_diff": round(float(np.median(a - b)), 4), "W": float(r.statistic), "p": float(f"{r.pvalue:.3g}"), "pairs": int(len(a))}


out = {}
# ---- LoRA multi-run summaries
L = {}
for f in sorted(glob.glob(f"{R}/lora_*_f*_s*.json")):
    d = json.load(open(f)); key = os.path.basename(f)[5:-5]
    L[key] = d
def pick(prefix, test, metric):
    ks = sorted(k for k in L if k.startswith(prefix))
    return ks, [L[k][test][metric] for k in ks]

summ = {}
for name, prefix, test in [("SA_strict_group_split", "sa_dedup_strict_f", "SA_dedup_test"),
                           ("MD_split_by_fish", "md_fish_phash_f", "MD_fish_test"),
                           ("SA_parasitic_heldout_known", "sa_openset_strict_f", "SA_open_test"),
                           ("SA_to_MD", "sa2md_strict_fx", "MD_all"),
                           ("MD_to_SA", "md2sa_strict_fx", "SA_shared")]:
    ks, f1 = pick(prefix, test, "macroF1_known")
    _, acc = pick(prefix, test, "acc_known")
    if ks:
        summ[name] = {"runs": ks, "macroF1": ms(f1), "acc": ms(acc)}
        au = [L[k][test].get("unknown_AUROC") for k in ks]
        if au[0]:
            summ[name]["unknown_AUROC"] = {s: ms([a[s] for a in au]) for s in au[0]}
out["lora"] = summ

# ---- T1: MD random vs by-fish (frozen), 25 paired fold-seed values
ym, gm = md.cls.values, md.group.values
rnd, grp = [], []
for seed in range(5):
    for (tr, te), (tr2, te2) in zip(StratifiedKFold(5, shuffle=True, random_state=seed).split(XM, ym),
                                    StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(XM, ym, gm)):
        rnd.append(f1_score(ym[te], LogisticRegression(max_iter=3000).fit(XM[tr], ym[tr]).predict(XM[te]), average="macro"))
        grp.append(f1_score(ym[te2], LogisticRegression(max_iter=3000).fit(XM[tr2], ym[tr2]).predict(XM[te2]), average="macro"))
out["T1_MD_random_vs_fish_frozen"] = {"random": ms(rnd), "by_fish": ms(grp), "wilcoxon": wx(rnd, grp)}

# ---- T2: frozen vs LoRA on the same folds (seed-0 fold assignment as in train_lora.py)
def frozen_fold_scores(X, y, g, uniq=None):
    sc = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y, y, g):
        if uniq is not None:
            te = np.intersect1d(te, uniq)
        p = LogisticRegression(max_iter=3000).fit(X[tr], y[tr]).predict(X[te])
        sc.append(f1_score(y[te], p, average="macro"))
    return sc
sa_u = sa.groupby("group_strict").head(1).index.values
if "SA_strict_group_split" in summ and summ["SA_strict_group_split"]["macroF1"]["n"] == 5:
    fz = frozen_fold_scores(XS, sa.cls.values, sa.group_strict.values, sa_u)
    out["T2_SA_frozen_vs_lora"] = {"frozen": ms(fz), "lora": summ["SA_strict_group_split"]["macroF1"],
                                   "wilcoxon": wx(summ["SA_strict_group_split"]["macroF1"]["values"], fz)}
if "MD_split_by_fish" in summ and summ["MD_split_by_fish"]["macroF1"]["n"] == 5:
    fz = frozen_fold_scores(XM, ym, gm)
    out["T2_MD_frozen_vs_lora"] = {"frozen": ms(fz), "lora": summ["MD_split_by_fish"]["macroF1"],
                                   "wilcoxon": wx(summ["MD_split_by_fish"]["macroF1"]["values"], fz)}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{R}/v2_aggregate.json", "w"), indent=1)
