"""Stricter duplicate groups for the South Asia set.

Union of (a) the pHash groups from audit_southasia.py and (b) links between any two
images whose frozen DINOv2-S cosine similarity is >= THR (crops, zooms, flips of one
photograph). Writes column 'group_strict' into results/southasia_meta.csv.
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import json, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")
R = _ROOT + "/results"
THR = 0.85
meta = pd.read_csv(f"{R}/southasia_meta.csv")
X = normalize(np.load(f"{R}/feat_southasia_dinov2s.npy"))
n = len(meta)
parent = list(range(n))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


for g, idx in meta.groupby("group").indices.items():
    for j in idx[1:]:
        union(idx[0], j)
S = X @ X.T
iu = np.argwhere(np.triu(S, 1) >= THR)
for i, j in iu:
    union(int(i), int(j))
meta["group_strict"] = [find(i) for i in range(n)]

sizes = meta.groupby("group_strict").size()
mixed = meta.groupby("group_strict").label.nunique()
out = {"threshold": THR, "cosine_links": int(len(iu)),
       "groups_phash": int(meta.group.nunique()), "groups_strict": int(meta.group_strict.nunique()),
       "largest_group": int(sizes.max()), "groups_with_mixed_labels": int((mixed > 1).sum()),
       "images_in_mixed_groups": int(meta.group_strict.map(mixed).gt(1).sum()),
       "groups_per_class": meta.groupby("label").group_strict.nunique().to_dict()}
print(json.dumps(out, indent=1, default=int))
meta.to_csv(f"{R}/southasia_meta.csv", index=False)
json.dump(out, open(f"{R}/regroup_strict.json", "w"), indent=1, default=int)
