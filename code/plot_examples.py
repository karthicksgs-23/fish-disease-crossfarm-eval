"""Example figures: study workflow, model block diagram, sample images,
leaked duplicate pairs, and example cross-farm predictions.

fig0_workflow.png      flow diagram of the whole study
fig0_model.png         block diagram of DINOv2-S + LoRA with the scoring heads
fig0_samples.png       sample photographs, (a) South Asia 7 classes, (b) MatsyaDx-BD 4 classes
fig0_duplicates.png    train/test pairs from the official South Asia split
fig0_predictions.png   MatsyaDx-BD photos scored by the LoRA model trained on South Asia (seed 0)
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageOps
from scipy.special import softmax
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")
ROOT = _ROOT + ""
R = ROOT + "/results"; FIG = ROOT + "/figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
SA_NAMES = {0: "Bacterial red", 1: "Aeromoniasis", 2: "Bacterial gill", 3: "Saprolegniasis",
            4: "Healthy", 5: "Parasitic", 6: "Viral white-tail"}
MD_NAMES = {"Bacterial Gill Diseases": "Bacterial gill", "Bacterial Red Diseases": "Bacterial red",
            "Epizootic Ulcerative Syndrome (EUS)": "EUS", "Healthy Fish": "Healthy"}
SPECIES = {"Grass_carp_(Ctenopharyngodon idella)": "grass carp", "Rui_(Labeo rohita)": "rohu",
           "Silver_carp_(Hypophthalmichthys molitrix)": "silver carp"}
sa = pd.read_csv(f"{R}/southasia_meta.csv"); md = pd.read_csv(f"{R}/matsyadx_meta.csv")
STOCK = sa.group_strict.value_counts().index[0]          # 75-image stock-photo group: never shown


def load(path, size=256):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((size, size))
    return ImageOps.pad(im, (size, size), color=(255, 255, 255))


def show(ax, img, title=None, color="black", fs=8):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#999999"); s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=fs, color=color, pad=3)


# ------------------------------------------------------------ boxes and arrows
def box(ax, x, y, w, h, text, fc="#eef3fa", ec="#2f5d8a", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color="#444444"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.0, color=color,
                                 shrinkA=0, shrinkB=0))


def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 7.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    C_DATA, C_AUD, C_SPL, C_MOD, C_EVA = "#e8f1fb", "#fff4e0", "#eaf6ea", "#f3ecf8", "#fdecec"
    # row 1: data
    box(ax, 0.27, 0.94, 0.40, 0.08, "South Asia set (Kaggle)\n7 classes, 2,450 images, 128 × 128", C_DATA)
    box(ax, 0.73, 0.94, 0.40, 0.08, "MatsyaDx-BD v2 (Bangladesh farms)\n4 classes, 2,137 photos, 86 fish", C_DATA)
    # row 2: audit
    box(ax, 0.27, 0.80, 0.40, 0.10, "Duplicate audit\nMD5 exact copies → pHash (≤ 6 bits)\n→ DINOv2 cosine ≥ 0.85 → 1,176 groups", C_AUD)
    box(ax, 0.73, 0.80, 0.40, 0.10, "Fish identity from metadata\n(every photo linked to one of 86 fish)", C_AUD)
    arrow(ax, (0.27, 0.90), (0.27, 0.85)); arrow(ax, (0.73, 0.90), (0.73, 0.85))
    # row 3: split protocols
    box(ax, 0.50, 0.655, 0.98, 0.10,
        "Split protocols   P1 official split   P2 official, copy-free test (124)   P3 grouped 5-fold\n"
        "P4 MatsyaDx random vs by fish   P5 train on one dataset, test on the other   P6 parasitic class held out",
        C_SPL, fs=7.2)
    arrow(ax, (0.27, 0.75), (0.27, 0.705)); arrow(ax, (0.73, 0.75), (0.73, 0.705))
    # row 4: models
    box(ax, 0.27, 0.51, 0.40, 0.10, "Frozen features\nDINOv2 ViT-S/14 (384-d), ResNet-50 (2,048-d)\n+ logistic regression", C_MOD)
    box(ax, 0.73, 0.51, 0.40, 0.10, "Fine-tuned DINOv2 ViT-S/14\nLoRA rank 8 on qkv + proj, 12 blocks\n8 epochs, AdamW, CPU", C_MOD)
    arrow(ax, (0.27, 0.605), (0.27, 0.56)); arrow(ax, (0.73, 0.605), (0.73, 0.56))
    # row 5: four evaluations
    xs = [0.125, 0.375, 0.625, 0.875]
    texts = ["Closed-set\naccuracy,\nmacro-F1", "Cross-farm\nshift + remedies\n(centering, CORAL,\nlabelled fish)",
             "Unknown disease\nMSP, energy,\nkNN, Mahalanobis\n(AUROC, FPR95)", "Conformal sets\nphoto-level vs\nfish-level\n(coverage, size)"]
    for x, t in zip(xs, texts):
        box(ax, x, 0.32, 0.225, 0.14, t, C_EVA, fs=7.6)
    ax.plot([0.125, 0.875], [0.425, 0.425], color="#444444", lw=1.0)
    ax.plot([0.27, 0.27], [0.46, 0.425], color="#444444", lw=1.0); ax.plot([0.73, 0.73], [0.46, 0.425], color="#444444", lw=1.0)
    for x in xs:
        arrow(ax, (x, 0.425), (x, 0.39))
    # row 6: statistics + output
    box(ax, 0.50, 0.15, 0.92, 0.10, "Reporting: mean ± SD over folds / seeds, paired Wilcoxon signed-rank tests\n"
        "→ leakage-free estimate of how a fish-disease classifier behaves on a new farm", "#f2f2f2", ec="#555555", fs=8)
    for x in xs:
        arrow(ax, (x, 0.25), (x, 0.20))
    fig.savefig(f"{FIG}/fig0_workflow.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_model():
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    fig, ax = plt.subplots(figsize=(7.6, 3.5)); ax.set_xlim(0, 1.12); ax.set_ylim(0, 1); ax.axis("off")
    row = md[(md.health_condition == "Bacterial Red Diseases")].iloc[5]
    img = np.asarray(load(row.path, 220))
    ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.21), (0.06, 0.55), frameon=False))
    ax.text(0.065, 0.30, "Input photo\nletterboxed\n224 × 224", ha="center", va="center", fontsize=7.3)
    box(ax, 0.215, 0.55, 0.11, 0.28, "Patch\nembedding\n16 × 16\npatches\n+ [CLS]", "#e8f1fb", fs=7.3)
    ax.add_patch(FancyBboxPatch((0.30, 0.12), 0.30, 0.76, boxstyle="round,pad=0.01", fc="#f7f7f7", ec="#777777", lw=0.8, ls="--"))
    ax.text(0.45, 0.92, "12 × transformer blocks (weights frozen)", ha="center", fontsize=7.5)
    box(ax, 0.45, 0.76, 0.27, 0.12, "Multi-head self-attention\nqkv: W x + B A x  (LoRA)", "#f3ecf8", fs=7.2)
    box(ax, 0.45, 0.58, 0.27, 0.10, "Output projection\nW x + B A x  (LoRA)", "#f3ecf8", fs=7.2)
    box(ax, 0.45, 0.41, 0.27, 0.10, "LayerNorm + MLP (frozen)", "#ffffff", fs=7.2)
    box(ax, 0.45, 0.22, 0.27, 0.10, "r = 8, α = 16:\n0.22 M of 22 M weights trained", "#fff4e0", ec="#b07a1a", fs=7)
    arrow(ax, (0.45, 0.70), (0.45, 0.635)); arrow(ax, (0.45, 0.53), (0.45, 0.46))
    arrow(ax, (0.122, 0.55), (0.158, 0.55)); arrow(ax, (0.272, 0.55), (0.305, 0.55))
    box(ax, 0.68, 0.55, 0.10, 0.18, "[CLS]\nfeature f\n(384-d)", "#e8f1fb", fs=7.3)
    arrow(ax, (0.605, 0.55), (0.628, 0.55))
    box(ax, 0.84, 0.77, 0.14, 0.14, "Linear head\nlogits z,\nsoftmax p", "#eaf6ea", fs=7.2)
    box(ax, 0.84, 0.30, 0.15, 0.16, "Training\nfeatures, class\nmeans,\ncovariance", "#eaf6ea", fs=6.8)
    arrow(ax, (0.73, 0.60), (0.77, 0.74)); arrow(ax, (0.73, 0.50), (0.77, 0.34))
    box(ax, 1.03, 0.88, 0.15, 0.09, "Class prediction", "#fdecec", fs=7.0)
    box(ax, 1.03, 0.67, 0.15, 0.14, "MSP, energy\n(unknown);\nconformal set", "#fdecec", fs=7.0)
    box(ax, 1.03, 0.30, 0.15, 0.14, "kNN,\nMahalanobis\n(unknown)", "#fdecec", fs=7.0)
    arrow(ax, (0.91, 0.81), (0.955, 0.87)); arrow(ax, (0.91, 0.73), (0.955, 0.69)); arrow(ax, (0.91, 0.30), (0.955, 0.30))
    fig.savefig(f"{FIG}/fig0_model.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_samples():
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2.25], hspace=0.16)
    # (a) South Asia: 3 images per class, distinct strict groups, stock group excluded
    g1 = gs[0].subgridspec(3, 7, wspace=0.05, hspace=0.05)
    for c in range(7):
        pool = sa[(sa.label == c) & (sa.group_strict != STOCK)].groupby("group_strict").head(1)
        pick = pool.sample(3, random_state=int(rng.integers(1e6)))
        for r, (_, row) in enumerate(pick.iterrows()):
            ax = fig.add_subplot(g1[r, c]); show(ax, load(row.path, 160), SA_NAMES[c] if r == 0 else None, fs=7)
    fig.text(0.02, 0.89, "(a)", fontsize=10, fontweight="bold")
    # (b) MatsyaDx-BD: one photo per species per class, each from a different fish
    g2 = gs[1].subgridspec(2, 6, wspace=0.05, hspace=0.18)
    k = 0
    for cls in MD_NAMES:
        sub = md[md.health_condition == cls]
        for sp in list(SPECIES)[:3]:
            if k >= 12:
                break
            fish = sub[sub.fish_category == sp]
            if not len(fish):
                continue
            row = fish.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
            ax = fig.add_subplot(g2[k // 6, k % 6]); show(ax, load(row.path, 300), f"{MD_NAMES[cls]}\n{SPECIES[sp]}", fs=6.8)
            k += 1
    fig.text(0.02, 0.405, "(b)", fontsize=10, fontweight="bold")
    fig.savefig(f"{FIG}/fig0_samples.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def phash_bits(path):
    im = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    from scipy.fftpack import dct
    a = dct(dct(np.asarray(im, float), axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    return (a > np.median(a)).flatten()


def fig_duplicates():
    X = normalize(np.load(f"{R}/feat_southasia_dinov2s.npy"))
    tr, te = sa[sa.split == "train"], sa[sa.split == "test"]
    pairs = {"exact": [], "near": [], "feat": []}
    for i, row in te.iterrows():
        same_md5 = tr[tr.md5 == row.md5]
        if len(same_md5):
            pairs["exact"].append((same_md5.index[0], i)); continue
        same_g = tr[tr.group == row.group]
        if len(same_g):
            pairs["near"].append((same_g.index[0], i)); continue
        same_s = tr[tr.group_strict == row.group_strict]
        if len(same_s) and row.group_strict != STOCK:
            j = same_s.index[np.argmax(X[same_s.index] @ X[i])]
            pairs["feat"].append((j, i))
    rng = np.random.default_rng(7)
    chosen = []
    for kind, lab in [("exact", "Exact copy"), ("near", "pHash near-copy"), ("feat", "Feature link")]:
        p = pairs[kind]
        for t in rng.choice(len(p), 2, replace=False):
            a, b = p[t]
            hd = int((phash_bits(sa.path[a]) != phash_bits(sa.path[b])).sum())
            chosen.append((lab, a, b, hd, float(X[a] @ X[b])))
    print({k: len(v) for k, v in pairs.items()})
    fig, axes = plt.subplots(2, 6, figsize=(7.4, 3.0), gridspec_kw={"wspace": 0.06, "hspace": 0.05})
    for col, (lab, a, b, hd, cs) in enumerate(chosen):
        show(axes[0, col], load(sa.path[a], 160), f"{lab}\nΔ = {hd} bits, cos {cs:.2f}", fs=6.2)
        show(axes[1, col], load(sa.path[b], 160))
        axes[1, col].set_xlabel(SA_NAMES[sa.label[b]], fontsize=6.8)
    axes[0, 0].set_ylabel("Training", fontsize=8); axes[1, 0].set_ylabel("Test", fontsize=8)
    fig.savefig(f"{FIG}/fig0_duplicates.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return {k: len(v) for k, v in pairs.items()}, chosen


def fig_predictions():
    d = np.load(f"{R}/lora_sa2md_strict_fx_s0.npz", allow_pickle=True)
    classes = ["gill", "healthy", "red"]; nice = {"gill": "Bacterial gill", "healthy": "Healthy", "red": "Bacterial red", "eus": "EUS"}
    P = softmax(d["MD_all_logits"], 1); y = d["MD_all_y"]; pred = np.array(classes)[P.argmax(1)]
    conf = P.max(1)
    # the npz rows follow the order of the MatsyaDx-BD metadata file
    assert len(y) == len(md)
    rng = np.random.default_rng(11)
    rows = []
    for c in classes:                       # one correct, one wrong per shared class
        ok = np.where((y == c) & (pred == c))[0]; bad = np.where((y == c) & (pred != c))[0]
        rows += [("ok", rng.choice(ok)), ("bad", rng.choice(bad))]
    eus = np.where(y == "eus")[0]
    rows += [("eus", i) for i in rng.choice(eus, 2, replace=False)]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.3), gridspec_kw={"wspace": 0.06, "hspace": 0.32})
    for ax, (kind, i) in zip(axes.flat, rows):
        col = {"ok": "#1a7f37", "bad": "#c0392b", "eus": "#b07a1a"}[kind]
        mark = {"ok": "✓", "bad": "✗", "eus": "?"}[kind]
        show(ax, load(md.path[i], 300), f"{mark} true: {nice[y[i]]}\npred: {nice[pred[i]]} ({conf[i]:.2f})", color=col, fs=7)
    fig.savefig(f"{FIG}/fig0_predictions.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    acc = {c: round(float((pred[y == c] == c).mean()), 3) for c in classes}
    return acc, [(k, md.image_id[i]) for k, i in rows]


if __name__ == "__main__":
    import sys as _s
    only = _s.argv[1:]
    if not only or "model" in only: fig_workflow(); fig_model()
    if only: _s.exit()
    fig_samples()
    info = {"duplicates": fig_duplicates(), "predictions": fig_predictions()}
    print(json.dumps(info, indent=1, default=str))
