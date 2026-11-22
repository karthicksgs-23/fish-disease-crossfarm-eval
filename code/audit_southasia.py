"""Leakage audit of the South Asia fish-disease set (HF mirror of the Kaggle data).

Exports images to data/southasia/images/<split>/<label>/<idx>.png, then finds
exact duplicates (MD5 of decoded pixels) and near duplicates (perceptual hash,
Hamming distance <= THRESH) within and across the train/test splits.
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import hashlib, io, json, sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(_ROOT + "")
SRC = ROOT / "data" / "southasia"
OUT = ROOT / "results"
THRESH = 6  # pHash Hamming distance treated as "near duplicate"
NAMES = ["Bacterial Red disease", "Aeromoniasis", "Bacterial gill disease",
         "Saprolegniasis", "Healthy", "Parasitic", "White tail disease"]

rows = []
for split in ["train", "test"]:
    df = pd.read_parquet(SRC / f"{split}.parquet")
    for i, r in df.iterrows():
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        d = SRC / "images" / split / str(r["label"])
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{i:05d}.png"
        if not p.exists():
            img.save(p)
        rows.append(dict(split=split, idx=i, label=int(r["label"]), path=str(p),
                         w=img.width, h=img.height,
                         md5=hashlib.md5(np.asarray(img).tobytes()).hexdigest(),
                         phash=imagehash.phash(img, hash_size=8)))
meta = pd.DataFrame(rows)

# size summary
print("image size (w x h) median:", int(meta.w.median()), "x", int(meta.h.median()),
      "| min", meta.w.min(), "x", meta.h.min(), "| max", meta.w.max(), "x", meta.h.max())

# near-duplicate groups via union-find on pHash distance
H = np.array([h.hash.flatten() for h in meta.phash], dtype=np.uint8)
n = len(H)
parent = list(range(n))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a
pairs = []
for i in range(n):
    d = (H[i + 1:] != H[i]).sum(1)
    for j in np.nonzero(d <= THRESH)[0] + i + 1:
        pairs.append((i, int(j), int(d[j - i - 1])))
        parent[find(i)] = find(int(j))
meta["group"] = [find(i) for i in range(n)]

exact_dup = meta.duplicated("md5", keep="first").sum()
gsize = meta.groupby("group").size()
in_dup_groups = int((meta.group.map(gsize) > 1).sum())
uniq = meta.group.nunique()

cross = sum(1 for i, j, _ in pairs if meta.split[i] != meta.split[j])
test_leaky = meta[(meta.split == "test")].group.isin(meta[meta.split == "train"].group).sum()
label_conflict = sum(1 for g, s in meta.groupby("group") if s.label.nunique() > 1)

summary = dict(
    total_images=n,
    exact_duplicate_images=int(exact_dup),
    near_dup_pairs=len(pairs),
    images_in_near_dup_groups=in_dup_groups,
    unique_groups_after_dedup=int(uniq),
    train_test_near_dup_pairs=int(cross),
    test_images_with_near_dup_in_train=int(test_leaky),
    test_total=int((meta.split == "test").sum()),
    groups_with_conflicting_labels=int(label_conflict),
    phash_threshold=THRESH,
)
print(json.dumps(summary, indent=2))
per_class = meta.groupby("label").agg(images=("idx", "size"), unique_groups=("group", "nunique"))
per_class.index = [NAMES[i] for i in per_class.index]
print(per_class)

meta.drop(columns=["phash"]).to_csv(OUT / "southasia_meta.csv", index=False)
json.dump(summary, open(OUT / "southasia_audit.json", "w"), indent=2)
pd.DataFrame(pairs, columns=["i", "j", "hamming"]).to_csv(OUT / "southasia_dup_pairs.csv", index=False)
