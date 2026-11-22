"""Map the Hugging Face mirror of the South Asia set to the original Kaggle release.

1. Lists the files of the Kaggle dataset (public metadata API, no download or login needed).
2. Matches every mirror image to a Kaggle file by class folder, file name and byte size.
3. Checks whether Kaggle Test images are pixel copies of Kaggle Train images (MD5 of decoded pixels
   from results/southasia_meta.csv) and scores frozen-feature logistic regression on the Kaggle split.
Writes results/southasia_kaggle_map.csv and results/kaggle_original_split.json.
"""
import json, os, sys, urllib.parse, urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
API = "https://www.kaggle.com/api/v1/datasets/list/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia"
FOLDER = {0: "Bacterial Red disease", 1: "Bacterial diseases - Aeromoniasis", 2: "Bacterial gill disease",
          3: "Fungal diseases Saprolegniasis", 4: "Healthy Fish", 5: "Parasitic diseases",
          6: "Viral diseases White tail disease"}

# 1) Kaggle file list
files, tok = [], None
while True:
    d = json.load(urllib.request.urlopen(API + (f"?pageToken={urllib.parse.quote(tok)}" if tok else ""), timeout=60))
    files += [(f["name"], f["totalBytes"]) for f in d.get("datasetFiles", [])]
    tok = d.get("nextPageTokenNullable")
    if not tok:
        break
kag = defaultdict(list)                       # (class folder, file name) -> [(Train/Test, bytes)]
for name, size in files:
    p = name.split("/")
    if len(p) == 4:
        kag[(p[2], p[3])].append((p[1], size))
print("Kaggle image files:", sum(len(v) for v in kag.values()))

# 2) match mirror images
rows = []
for split in ["train", "test"]:
    df = pd.read_parquet(os.path.join(ROOT, "data", "southasia", f"{split}.parquet"))
    for _, r in df.iterrows():
        name, size = r.image["path"], len(r.image["bytes"])
        cands = kag.get((FOLDER[r.label], name), [])
        rows.append(dict(mirror_split=split, label=r.label, name=name, n_cand=len(cands),
                         n_size_match=sum(c[1] == size for c in cands)))
m = pd.DataFrame(rows)
print("mirror images matched by name and size:", int((m.n_size_match > 0).sum()), "of", len(m))

# 3) same-name pairs = one Kaggle Train copy + one Kaggle Test copy
meta = pd.read_csv(os.path.join(R, "southasia_meta.csv"))
assert (meta.split.values == m.mirror_split.values).all() and (meta.label.values == m.label.values).all()
m["md5"] = meta.md5.values
pairs = m[m.n_cand == 2].groupby(["label", "name"])
identical = int((pairs.md5.nunique() == 1).sum())
m["kaggle_split"] = "Train"
m.loc[pairs.head(1).index, "kaggle_split"] = "Test"   # the two copies are identical, so either can be the Test one
m.drop(columns="md5").to_csv(os.path.join(R, "southasia_kaggle_map.csv"), index=False)

tr = np.where(m.kaggle_split == "Train")[0]; te = np.where(m.kaggle_split == "Test")[0]
y = meta.label.values
out = {"n_train": int(len(tr)), "n_test": int(len(te)), "test_identical_to_train": identical}
for bb in ["dinov2s", "resnet50"]:
    X = normalize(np.load(os.path.join(R, f"feat_southasia_{bb}.npy")))
    p = LogisticRegression(max_iter=3000).fit(X[tr], y[tr]).predict(X[te])
    out[bb] = {"acc": round(accuracy_score(y[te], p), 4), "macroF1": round(f1_score(y[te], p, average="macro"), 4)}
out["test_sharing_strict_group_with_train"] = int(np.isin(meta.group_strict.values[te], meta.group_strict.values[tr]).sum())
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(R, "kaggle_original_split.json"), "w"), indent=1)
