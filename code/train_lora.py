"""LoRA fine-tuning of DINOv2 ViT-S/14 on CPU, one protocol per call.

usage: python train_lora.py <protocol> [--epochs N] [--bench]
protocols:
  sa_official  SA official train/test split (duplicates leak across)      [7 classes]
  sa_dedup     SA, fold 0 of StratifiedGroupKFold on near-duplicate groups [7 classes]
  md_fish      MatsyaDx, fold 0 of StratifiedGroupKFold by individual fish [4 classes]
  sa2md        train SA (shared classes) -> test MatsyaDx (shared + EUS as unknown)
  md2sa        train MatsyaDx (shared)   -> test SA unique images (shared)
  sa_openset   SA group split (30 % test), Parasitic held out of training
Writes results/lora_<protocol>.npz (logits, features, labels) and .json (metrics).
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import argparse, json, math, sys, time
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(8)
torch.manual_seed(0); np.random.seed(0)
R = _ROOT + "/results"
SA_NAMES = {0: "red", 1: "aeromoniasis", 2: "gill", 3: "saprolegniasis", 4: "healthy", 5: "parasitic", 6: "whitetail"}
MD_NAMES = {"Bacterial Gill Diseases": "gill", "Bacterial Red Diseases": "red",
            "Epizootic Ulcerative Syndrome (EUS)": "eus", "Healthy Fish": "healthy"}
SHARED = ["gill", "red", "healthy"]


# ---------------- data / protocols ----------------
def load_meta():
    sa = pd.read_csv(f"{R}/southasia_meta.csv"); sa["cls"] = sa.label.map(SA_NAMES)
    md = pd.read_csv(f"{R}/matsyadx_meta.csv"); md["cls"] = md.health_condition.map(MD_NAMES)
    return sa, md


def protocol(name, fold=0, gcol="group"):
    sa, md = load_meta()
    sa["group"] = sa[gcol]  # pHash groups or strict (pHash + DINOv2 cos>=0.85) groups
    Isa = np.load(f"{R}/cache_southasia_256.npy", mmap_mode="r")
    Imd = np.load(f"{R}/cache_matsyadx_256.npy", mmap_mode="r")
    sa_u = sa.groupby("group").head(1).index.values  # one image per near-duplicate group
    ysa, ymd = sa.cls.values, md.cls.values
    if name == "sa_official":
        tr = np.where(sa.split == "train")[0]; te = np.where(sa.split == "test")[0]
        trg = set(sa.group[tr]); clean = np.array([sa.group[i] not in trg for i in te])
        return dict(Xtr=Isa[tr], ytr=ysa[tr], tests={"SA_test_all": (Isa[te], ysa[te]),
                    "SA_test_clean": (Isa[te[clean]], ysa[te[clean]])})
    if name == "sa_dedup":
        tr, te = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(ysa, ysa, sa.group))[fold]
        te = np.intersect1d(te, sa_u)
        return dict(Xtr=Isa[tr], ytr=ysa[tr], tests={"SA_dedup_test": (Isa[te], ysa[te])})
    if name == "md_fish":
        tr, te = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(ymd, ymd, md.group))[fold]
        return dict(Xtr=Imd[tr], ytr=ymd[tr], tests={"MD_fish_test": (Imd[te], ymd[te])})
    if name == "sa2md":
        tr = np.array([i for i in sa_u if ysa[i] in SHARED])
        te = np.where(md.cls.isin(SHARED + ["eus"]))[0]
        return dict(Xtr=Isa[tr], ytr=ysa[tr], tests={"MD_all": (Imd[te], ymd[te])}, unknown="eus")
    if name == "md2sa":
        tr = np.where(md.cls.isin(SHARED))[0]
        te = np.array([i for i in sa_u if ysa[i] in SHARED])
        return dict(Xtr=Imd[tr], ytr=ymd[tr], tests={"SA_shared": (Isa[te], ysa[te])})
    if name == "sa_openset":
        if fold < 0:  # v1 pilot split (30 % of groups, GroupShuffleSplit)
            trg, teg = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(sa_u, groups=sa.group.values[sa_u]))
        else:         # five-fold group split over unique images
            trg, teg = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(sa_u, ysa[sa_u], sa.group.values[sa_u]))[fold]
        tr_groups = set(sa.group.values[sa_u[trg]])
        tr = np.array([i for i in range(len(sa)) if sa.group[i] in tr_groups and ysa[i] != "parasitic"])
        te = sa_u[teg]
        return dict(Xtr=Isa[tr], ytr=ysa[tr], tests={"SA_open_test": (Isa[te], ysa[te])}, unknown="parasitic")
    raise ValueError(name)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
AUG = v2.Compose([v2.RandomResizedCrop(224, scale=(0.6, 1.0), antialias=True), v2.RandomHorizontalFlip(),
                  v2.ColorJitter(0.3, 0.3, 0.3, 0.05)])


def to_tensor(batch_u8, train):
    x = torch.from_numpy(np.ascontiguousarray(batch_u8)).permute(0, 3, 1, 2)
    if train:
        x = torch.stack([AUG(i) for i in x])
    else:
        x = v2.functional.resize(x, [224, 224], antialias=True)
    return (x.float() / 255 - MEAN) / STD


# ---------------- LoRA ----------------
class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.s


def build(n_classes):
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
    for p in m.parameters():
        p.requires_grad = False
    for blk in m.blocks:
        blk.attn.qkv = LoRA(blk.attn.qkv)
        blk.attn.proj = LoRA(blk.attn.proj)
    head = nn.Linear(m.num_features, n_classes)
    return m, head


def embed_logits(m, head, X, bs=64):
    m.eval(); F_, L_ = [], []
    with torch.inference_mode():
        for i in range(0, len(X), bs):
            f = m(to_tensor(X[i:i + bs], False))
            F_.append(f); L_.append(head(f))
    return torch.cat(F_).numpy(), torch.cat(L_).numpy()


def unknown_scores(logits, f, ftr, ytr_i):
    from scipy.special import logsumexp, softmax
    fn = f / np.linalg.norm(f, axis=1, keepdims=True); trn = ftr / np.linalg.norm(ftr, axis=1, keepdims=True)
    return {"MSP": 1 - softmax(logits, 1).max(1), "energy": -logsumexp(logits, 1),
            "kNN": 1 - np.sort(fn @ trn.T, 1)[:, -10:].mean(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol"); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bench", action="store_true"); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--fold", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="group", choices=["group", "group_strict"])
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tag = "" if a.fold is None and a.groups == "group" and a.seed == 0 else         f"_{'strict' if a.groups == 'group_strict' else 'phash'}_f{a.fold if a.fold is not None else 'x'}_s{a.seed}"
    P = protocol(a.protocol, fold=(a.fold if a.fold is not None else (-1 if a.protocol == "sa_openset" else 0)), gcol=a.groups)
    classes = sorted(set(P["ytr"])); ci = {c: i for i, c in enumerate(classes)}
    ytr = np.array([ci[c] for c in P["ytr"]])
    Xtr = np.asarray(P["Xtr"])
    m, head = build(len(classes))
    params = [p for p in m.parameters() if p.requires_grad] + list(head.parameters())
    print(a.protocol, "train", len(ytr), "classes", classes, "trainable", sum(p.numel() for p in params), flush=True)
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.05)
    steps = a.epochs * math.ceil(len(ytr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.1)
    w = torch.tensor(len(ytr) / (len(classes) * np.bincount(ytr, minlength=len(classes))), dtype=torch.float32)
    t0 = time.time()
    for ep in range(a.epochs):
        m.train(); perm = np.random.permutation(len(ytr)); tot = 0
        for k, i in enumerate(range(0, len(perm), a.bs)):
            b = np.sort(perm[i:i + a.bs])
            x = to_tensor(Xtr[b], True); y = torch.from_numpy(ytr[b])
            loss = F.cross_entropy(head(m(x)), y, weight=w, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
            if a.bench and k == 4:
                dt = (time.time() - t0) / 5
                print(f"BENCH {dt:.2f}s/batch -> {dt * steps / a.epochs / 60:.1f} min/epoch", flush=True)
                return
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(ytr):.4f} elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    ftr, _ = embed_logits(m, head, Xtr)
    out, save = {"protocol": a.protocol, "classes": classes, "epochs": a.epochs, "n_train": int(len(ytr)),
                 "train_minutes": round((time.time() - t0) / 60, 1)}, {"ftr": ftr, "ytr": ytr}
    for tname, (X, y) in P["tests"].items():
        f, lg = embed_logits(m, head, np.asarray(X))
        pred = np.array(classes)[lg.argmax(1)]
        known = np.isin(y, classes)
        r = {"n": int(len(y)), "n_known": int(known.sum()),
             "acc_known": accuracy_score(y[known], pred[known]),
             "macroF1_known": f1_score(y[known], pred[known], average="macro")}
        if "unknown" in P and (~known).any():
            sc = unknown_scores(lg, f, ftr, ytr)
            r["unknown_AUROC"] = {k: roc_auc_score((~known).astype(int), v) for k, v in sc.items()}
        out[tname] = r
        save[f"{tname}_f"], save[f"{tname}_logits"], save[f"{tname}_y"] = f, lg, y
    out.update({"fold": a.fold, "seed": a.seed, "groups": a.groups, "batch_size": a.bs})
    np.savez_compressed(f"{R}/lora_{a.protocol}{tag}.npz", **save)
    json.dump(out, open(f"{R}/lora_{a.protocol}{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
