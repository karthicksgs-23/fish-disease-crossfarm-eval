"""Residual-leakage check for the sa_dedup LoRA result.

For every test image of the sa_dedup fold, find its most similar training image
(cosine similarity of frozen DINOv2-S features). High similarity = likely a crop /
shift of the same source photo that the pHash threshold missed. Report LoRA
accuracy separately for test images with and without such a "twin".
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")
R = _ROOT + "/results"
SA_NAMES = {0: "red", 1: "aeromoniasis", 2: "gill", 3: "saprolegniasis", 4: "healthy", 5: "parasitic", 6: "whitetail"}
sa = pd.read_csv(f"{R}/southasia_meta.csv"); sa["cls"] = sa.label.map(SA_NAMES)
ysa = sa.cls.values
sa_u = sa.groupby("group").head(1).index.values
tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(ysa, ysa, sa.group))  # same as train_lora.py
te = np.intersect1d(te, sa_u)

X = normalize(np.load(f"{R}/feat_southasia_dinov2s.npy"))
sim = X[te] @ X[tr].T
best = sim.max(1); nn = tr[sim.argmax(1)]
same_label = ysa[nn] == ysa[te]

z = np.load(f"{R}/lora_sa_dedup.npz", allow_pickle=True)
classes = json.load(open(f"{R}/lora_sa_dedup.json"))["classes"]
pred = np.array(classes)[z["SA_dedup_test_logits"].argmax(1)]
assert (z["SA_dedup_test_y"] == ysa[te]).all(), "test order mismatch"
correct = pred == ysa[te]

out = {"n_test": int(len(te)), "lora_acc_all": round(float(correct.mean()), 4),
       "nearest_train_similarity_quantiles": {q: round(float(np.quantile(best, q)), 3) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}}
for t in [0.80, 0.85, 0.90, 0.95]:
    tw = best >= t
    out[f"twin_sim>={t}"] = {"test_with_twin": int(tw.sum()), "share": round(float(tw.mean()), 3),
                             "twin_same_label": round(float(same_label[tw].mean()), 3) if tw.any() else None,
                             "lora_acc_with_twin": round(float(correct[tw].mean()), 4) if tw.any() else None,
                             "lora_acc_without_twin": round(float(correct[~tw].mean()), 4) if (~tw).any() else None,
                             "n_without_twin": int((~tw).sum())}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{R}/check_residual_leak.json", "w"), indent=1)

# contact sheet: 8 test images with their nearest training image, sim 0.85-0.95
from PIL import Image
idx = np.where((best >= 0.85) & (best < 0.95))[0][:8]
W = 128; im = Image.new("RGB", (W * 4 + 30, W * 4), "white")
for k, i in enumerate(idx):
    r, c = k % 4, k // 4
    for t, p in enumerate([sa.path[te[i]], sa.path[nn[i]]]):
        im.paste(Image.open(p).convert("RGB").resize((W, W)), (c * (2 * W + 30) + t * W, r * W))
im.save(f"{R}/residual_twins_check.png")
print("sims shown:", [round(float(best[i]), 3) for i in idx])
